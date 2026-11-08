"""
DocuChat: a beginner-friendly RAG (Retrieval-Augmented Generation) app.

How it works:
  1. Read the PDF. Text is extracted page by page. Pictures (diagrams, charts)
     are described by Gemini, and the description is stored as extra text.
  2. Split the text into small overlapping chunks.
  3. Store the chunks in a vector database (ChromaDB), which turns each chunk
     into an "embedding" so we can search by meaning, not just keywords.
  4. When you ask a question, find the chunks most similar to it.
  5. Send those chunks plus your question to Gemini and ask it to answer
     using ONLY that context, citing page numbers.

Image descriptions are saved in a small file next to this script, so each
image is only described once. Re-uploading the same book is fast.
"""

import hashlib
import json
import time
import uuid
from pathlib import Path

import chromadb
import streamlit as st
from google import genai
from google.genai import errors, types
from pypdf import PdfReader

# ---------- Settings you can experiment with ----------
CHUNK_SIZE = 1200     # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks
TOP_K = 10            # how many chunks to retrieve per question
# Free Gemini model. Check current names at aistudio.google.com
MODEL = "gemini-3.8-flash"
# Tried in order if the main model is overloaded or unavailable.
FALLBACK_MODELS = ["gemini-3.7-flash", "gemini-3.5-flash-lite", "gemini-flash-lite-latest"]

# Reading pictures (diagrams, charts) inside the PDF with Gemini
MAX_IMAGES = 30            # safety limit so the free tier isn't used up
MIN_IMAGE_BYTES = 15000    # skip tiny images such as icons and decorations
IMAGE_PAUSE_SECONDS = 4    # pause between image requests (free tier rate limit)
IMAGE_TIME_BUDGET_SECONDS = 180  # stop asking Gemini about new images after this long
MAX_IMAGE_FAILURES = 3     # stop image reading after this many failures in a row
IMAGE_PHASE_LIMIT_SECONDS = 300  # hard limit for ALL image work (decoding + Gemini)
IMAGE_REQUEST_TIMEOUT = 30       # give up on one image request after this many seconds
ANSWER_REQUEST_TIMEOUT = 40      # give up on one answer request after this many seconds
CACHE_FILE = Path(__file__).parent / ".docuchat_cache.json"


# ---------- Small helpers ----------
def make_client(timeout_seconds):
    """Create a Gemini client that gives up on a request after `timeout_seconds`."""
    try:
        options = types.HttpOptions(timeout=int(timeout_seconds * 1000))  # milliseconds
        return genai.Client(http_options=options)
    except Exception:
        return genai.Client()  # older library versions: fall back to the default


def request_config(system_prompt, timeout_seconds):
    """Build the settings for one request, with a time limit when the library allows it."""
    try:
        return types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=800,
            http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        )
    except Exception:
        return types.GenerateContentConfig(
            system_instruction=system_prompt, max_output_tokens=800
        )


def parse_page_range(text):
    """Turn '15-25' (or '20') into (15, 25). Returns None if empty or not valid."""
    text = (text or "").strip().replace("–", "-")
    if not text:
        return None
    try:
        if "-" in text:
            first, last = (int(part) for part in text.split("-", 1))
        else:
            first = last = int(text)
    except ValueError:
        return None
    if first < 1 or last < first:
        return None
    return first, last


# ---------- Saving image descriptions so we never repeat the slow step ----------
def load_cache():
    """Load previously saved image descriptions (empty if there are none)."""
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(cache):
    """Save descriptions to disk. A failure here must never break the app."""
    try:
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


# ---------- Step 1: read the PDF (text and images) ----------
def describe_image(client, image_bytes, mime_type, cache):
    """Ask Gemini to transcribe and explain one image, and save the result.

    Returns the description, which is '' for decorative images. If Gemini is busy
    this fails fast (one try, one backup model) so the upload doesn't crawl.
    """
    key = hashlib.sha256(image_bytes).hexdigest()

    prompt = (
        "This image comes from a book. If it is a diagram, chart, table or figure, "
        "transcribe ALL the text in it and explain what it shows, naming every "
        "labelled part and how the parts relate. If it is only decorative "
        "(a photo, ornament, or background), reply with exactly: DECORATIVE"
    )
    image = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    answer = call_gemini(
        client,
        "You describe images from books accurately.",
        [image, prompt],
        max_attempts=1,
        models=[MODEL] + FALLBACK_MODELS[:1],
        timeout_seconds=IMAGE_REQUEST_TIMEOUT,
    )
    answer = (answer or "").strip()
    description = "" if answer.upper().startswith("DECORATIVE") else answer

    cache[key] = description
    save_cache(cache)  # saved right away, so an interrupted upload loses nothing
    return description


def image_mime_type(name):
    """Return the image type from its file name, or None if we can't use it."""
    name = name.lower()
    if name.endswith(".png"):
        return "image/png"
    if name.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    return None


def extract_pages(pdf_file, client=None, progress=None, notes=None,
                  image_pages=None, timings=None):
    """Return a list of (page_number, text). Image descriptions become extra entries.

    image_pages: (first, last) page range to scan for images, or None for all pages.
    Image reading stops early if it takes too long or Gemini keeps failing.
    Why it stopped is added to `notes`, and the text of the book is always indexed.
    `timings` (a dict) is filled in with how long each part took.
    """
    reader = PdfReader(pdf_file)
    cache = load_cache()
    pages = []
    images_seen = 0
    image_seconds = 0.0     # time spent on ALL image work (decoding + Gemini)
    gemini_seconds = 0.0    # time spent waiting on Gemini for new images
    text_seconds = 0.0
    new_images = 0          # images sent to Gemini this time
    saved_images = 0        # images reused from the saved file
    failures = 0            # image requests that failed in a row
    stopped = None          # "time" or "busy" if we gave up on images early
    total = len(reader.pages)

    for i, page in enumerate(reader.pages, start=1):
        t = time.monotonic()
        text = page.extract_text() or ""
        text_seconds += time.monotonic() - t
        if text.strip():
            pages.append((i, text))

        in_range = image_pages is None or image_pages[0] <= i <= image_pages[1]
        if client is not None and in_range and stopped is None and images_seen < MAX_IMAGES:
            block_start = time.monotonic()
            try:
                page_images = list(page.images)
            except Exception:
                page_images = []  # some PDFs have images pypdf can't decode
            for img in page_images:
                if images_seen >= MAX_IMAGES or stopped:
                    break
                if image_seconds + (time.monotonic() - block_start) > IMAGE_PHASE_LIMIT_SECONDS:
                    stopped = "time"
                    break
                try:
                    mime = image_mime_type(img.name)
                    data = img.data
                    if mime is None or len(data) < MIN_IMAGE_BYTES:
                        continue
                except Exception:
                    continue  # one bad image shouldn't stop the whole upload
                images_seen += 1

                key = hashlib.sha256(data).hexdigest()
                if key in cache:                       # already described before: free
                    description = cache[key]
                    saved_images += 1
                else:
                    if gemini_seconds > IMAGE_TIME_BUDGET_SECONDS:
                        stopped = "time"
                        break
                    started = time.monotonic()
                    try:
                        description = describe_image(client, data, mime, cache)
                        failures = 0
                    except Exception:
                        failures += 1
                        if failures >= MAX_IMAGE_FAILURES:
                            stopped = "busy"
                        gemini_seconds += time.monotonic() - started
                        continue
                    new_images += 1
                    time.sleep(IMAGE_PAUSE_SECONDS)
                    gemini_seconds += time.monotonic() - started

                if description:
                    pages.append((i, f"[Image or diagram on this page]\n{description}"))
            image_seconds += time.monotonic() - block_start

        if progress is not None:
            progress.progress(i / total, text=f"Reading page {i} of {total}")

    if notes is not None and stopped == "time":
        notes.append(
            "Image reading paused to keep the upload quick. The images read so far are "
            "saved. Click 'Read more images' to continue, or type the page range of the "
            "diagram you want (for example 15-25) and upload again."
        )
    elif notes is not None and stopped == "busy":
        notes.append(
            "Gemini was busy or unavailable, so image reading stopped early. The text of "
            "the book is fully indexed. Click 'Read more images' to try again later."
        )
    if timings is not None:
        timings.update(
            text_seconds=text_seconds,
            image_seconds=image_seconds,
            new_images=new_images,
            saved_images=saved_images,
        )
    return pages


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping pieces so ideas aren't cut off mid-way."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


# ---------- Step 2: store chunks in a vector database ----------
def build_index(pdf_file, read_images=True, progress=None, notes=None,
                image_pages=None, timings=None):
    """Chunk the PDF and put every chunk into a fresh ChromaDB collection."""
    client = chromadb.Client()  # in-memory, nothing saved to disk
    collection = client.create_collection(name=f"doc_{uuid.uuid4().hex[:8]}")
    gemini = make_client(IMAGE_REQUEST_TIMEOUT) if read_images else None

    ids, docs, metas = [], [], []
    pages = extract_pages(pdf_file, gemini, progress, notes, image_pages, timings)
    for page_num, text in pages:
        for chunk in chunk_text(text):
            ids.append(f"chunk_{len(ids)}")
            docs.append(chunk)
            metas.append({"page": page_num})

    if not docs:
        return None, 0

    # Chroma creates the embeddings for us using a built-in model.
    started = time.monotonic()
    if progress is not None:
        progress.progress(1.0, text="Building the search index...")
    collection.add(ids=ids, documents=docs, metadatas=metas)
    if timings is not None:
        timings["index_seconds"] = time.monotonic() - started
    return collection, len(docs)


# ---------- Step 3 + 4: retrieve and answer ----------
def answer_question(question, collection, models=None, info=None):
    results = collection.query(query_texts=[question], n_results=TOP_K)
    chunks = results["documents"][0]
    pages = [m["page"] for m in results["metadatas"][0]]

    context = "\n\n".join(
        f"[Page {p}]\n{c}" for p, c in zip(pages, chunks)
    )

    system_prompt = (
        "You answer questions using ONLY the document excerpts provided. "
        "If the answer is not in the excerpts, say you couldn't find it in the "
        "document. Mention the page number(s) your answer comes from."
    )
    user_prompt = f"Excerpts:\n{context}\n\nQuestion: {question}"

    client = make_client(ANSWER_REQUEST_TIMEOUT)  # reads GEMINI_API_KEY from your environment
    text = call_gemini(client, system_prompt, user_prompt, models=models, info=info)
    return text, sorted(set(pages)), chunks


def call_gemini(client, system_prompt, user_prompt, max_attempts=3, models=None,
                timeout_seconds=ANSWER_REQUEST_TIMEOUT, info=None):
    """Call Gemini, retrying on temporary errors and falling back to other models.

    If `info` (a dict) is given, it is filled in with the model that answered and
    how many seconds the whole call took.
    """
    began = time.monotonic()
    last_error = None
    for model in (models or [MODEL] + FALLBACK_MODELS):
        for attempt in range(max_attempts):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=request_config(system_prompt, timeout_seconds),
                )
                if info is not None:
                    info["model"] = model
                    info["seconds"] = time.monotonic() - began
                return response.text
            except errors.ServerError as e:      # 500/503: Google is busy, wait and retry
                last_error = e
                if attempt < max_attempts - 1:
                    time.sleep(2 ** (attempt + 1))   # waits 2s, 4s, 8s
            except errors.ClientError as e:
                if e.code in (404, 429):         # model missing or quota hit: try next model
                    last_error = e
                    break
                raise                            # anything else (bad key, etc.) is a real bug
    raise last_error


# ---------- The web interface ----------
st.set_page_config(page_title="DocuChat", page_icon="📄")
st.title("📄 DocuChat")
st.write("Upload a PDF, ask a question, and get an answer with the pages it came from.")

read_images = st.checkbox(
    "Read images and diagrams (slower the first time, fast after that)", value=True
)
page_range_text = ""
if read_images:
    page_range_text = st.text_input(
        "Pages to scan for images (optional, for example 15-25). "
        "Leave empty to scan the whole book, up to the time limit."
    )
image_pages = parse_page_range(page_range_text)
if page_range_text.strip() and image_pages is None:
    st.warning("I couldn't read that page range. Use a format like 15-25, or leave it empty.")
    st.stop()
uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    # Only rebuild the index when a new file (or a new image setting) is chosen.
    index_key = f"{uploaded.name}|{read_images}|{image_pages}"
    if st.session_state.get("index_key") != index_key:
        if read_images:
            st.caption(
                "Reading the PDF and describing its images. This can take a few minutes the "
                "first time and stops by itself after a time limit. Descriptions are saved, "
                "so the next upload is quick."
            )
        progress_bar = st.progress(0.0, text="Starting...")
        notes = []
        timings = {}
        collection, n_chunks = build_index(
            uploaded, read_images, progress_bar, notes, image_pages, timings
        )
        progress_bar.empty()
        if collection is None:
            st.error("No readable text found. Scanned PDFs need OCR first.")
            st.stop()
        st.session_state.collection = collection
        st.session_state.index_key = index_key
        st.session_state.n_chunks = n_chunks
        st.session_state.notes = notes
        st.session_state.timings = timings

    st.success(f"Indexed {st.session_state.n_chunks} chunks from {uploaded.name}")
    t = st.session_state.get("timings") or {}
    if t:
        st.caption(
            f"Time taken: reading text {t.get('text_seconds', 0):.0f}s, "
            f"images {t.get('image_seconds', 0):.0f}s "
            f"({t.get('new_images', 0)} new, {t.get('saved_images', 0)} from saved), "
            f"search index {t.get('index_seconds', 0):.0f}s."
        )
    for note in st.session_state.get("notes", []):
        st.info(note)
    if st.session_state.get("notes") and st.button("Read more images"):
        st.session_state.index_key = None  # forces a fresh index that reuses saved images
        st.rerun()

    all_models = [MODEL] + FALLBACK_MODELS
    answer_model = st.selectbox(
        "Answer model (if answers are slow, try one with 'lite' in the name)", all_models
    )
    model_order = [answer_model] + [m for m in all_models if m != answer_model]

    question = st.text_input("Ask a question about the document")
    if question:
        info = {}
        try:
            with st.spinner("Thinking... (up to a minute if Gemini is busy)"):
                reply, pages, chunks = answer_question(
                    question, st.session_state.collection, model_order, info
                )
        except Exception as e:  # busy, over the limit, timed out, or a bad key
            st.error(
                "Gemini couldn't answer right now (it may be busy, slow or over the free "
                f"limit). Wait a minute and ask again.\n\nDetails: {e}"
            )
            st.stop()
        st.markdown(reply)
        st.caption(f"Sources: pages {', '.join(map(str, pages))}")
        st.caption(f"Answered by {info.get('model', '?')} in {info.get('seconds', 0):.0f}s.")
        with st.expander("See the text the AI was given"):
            for c in chunks:
                st.text(c)
                st.divider()
else:
    st.info("Upload a PDF to get started.")
