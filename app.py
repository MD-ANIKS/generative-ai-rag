"""
Enterprise RAG Assistant: Streamlit + LangChain + Mistral + ChromaDB (in-memory)

Run:  streamlit run app.py
"""
import os
import tempfile
import uuid
from typing import List
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("USER_AGENT", "rag-streamlit-app/1.0")  # silences WebBaseLoader warning

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_KWARGS = {"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
EMBED_BATCH_SIZE = 32
DEFAULT_CHAT_MODEL = os.getenv("MISTRAL_CHAT_MODEL", "labs-leanstral-1-5")
NOT_FOUND_MSG = "I could not find the answer in the provided documents or links."

SYSTEM_PROMPT = f"""You are a strict, accurate AI assistant.
Answer ONLY using the provided context. Do not use outside knowledge and do not guess.
If the answer is not present in the context, reply with exactly this phrase and nothing else:
"{NOT_FOUND_MSG}"
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

st.set_page_config(page_title="RAG Assistant", page_icon="📚", layout="wide")

# Light styling that works in both light and dark themes (inherits theme colors)
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 960px;}
      [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.25);}
      .stat-pill {display:inline-block; padding:2px 10px; margin:2px 6px 2px 0;
                  border-radius:999px; border:1px solid rgba(128,128,128,.35); font-size:.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_embeddings() -> MistralAIEmbeddings:
    return MistralAIEmbeddings(model="mistral-embed")


@st.cache_resource(show_spinner=False)
def get_llm(model_name: str) -> ChatMistralAI:
    return ChatMistralAI(model=model_name, temperature=0, streaming=True)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state() -> None:
    defaults = {
        "messages": [],          # [{"role", "content", "sources"}]
        "vectorstore": None,
        "collection_name": None,
        "stats": None,           # {"files": n, "urls": n, "chunks": n}
        "uploader_key": 0,       # bump to reset the file uploader widget
        "urls_key": 0,           # bump to reset the URL text area
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def drop_vectorstore() -> None:
    """Delete the current Chroma collection (frees memory, avoids collisions)."""
    vs = st.session_state.get("vectorstore")
    if vs is not None:
        try:
            vs.delete_collection()
        except Exception:
            pass
    st.session_state.vectorstore = None
    st.session_state.collection_name = None
    st.session_state.stats = None


def clear_session() -> None:
    drop_vectorstore()
    st.session_state.messages = []
    st.session_state.uploader_key += 1
    st.session_state.urls_key += 1


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def is_valid_url(url: str) -> bool:
    p = urlparse(url)
    return p.scheme in ("http", "https") and bool(p.netloc)


def parse_urls(raw: str) -> List[str]:
    urls = [u.strip() for u in raw.replace(",", "\n").splitlines() if u.strip()]
    return list(dict.fromkeys(urls))  # de-duplicate, keep order


def load_pdfs(files) -> tuple[List[Document], List[str]]:
    docs, warnings = [], []
    for f in files:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.getvalue())
                tmp_path = tmp.name
            pages = PyPDFLoader(tmp_path).load()
            pages = [p for p in pages if p.page_content.strip()]
            if not pages:
                warnings.append(f"⚠️ **{f.name}**: no extractable text (empty or scanned PDF) – skipped.")
                continue
            for p in pages:
                p.metadata["source"] = f.name  # show real filename, not temp path
            docs.extend(pages)
        except Exception as e:
            warnings.append(f"⚠️ **{f.name}**: could not be read ({e}) – skipped.")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
    return docs, warnings


def load_urls(urls: List[str]) -> tuple[List[Document], List[str]]:
    docs, warnings = [], []
    for url in urls:
        if not is_valid_url(url):
            warnings.append(f"⚠️ **{url}**: invalid URL (must start with http:// or https://) – skipped.")
            continue
        try:
            pages = WebBaseLoader(url, requests_kwargs={"timeout": 20}).load()
            pages = [p for p in pages if p.page_content.strip()]
            if not pages:
                warnings.append(f"⚠️ **{url}**: no readable text found – skipped.")
                continue
            docs.extend(pages)
        except Exception as e:
            warnings.append(f"⚠️ **{url}**: failed to load ({type(e).__name__}) – skipped.")
    return docs, warnings


def build_index(files, urls: List[str]) -> None:
    """Load -> chunk -> embed -> store, with progress feedback."""
    drop_vectorstore()  # replace any previous index
    progress = st.sidebar.progress(0, text="Starting…")
    all_warnings: List[str] = []

    try:
        # 1) Loading
        with st.sidebar.status("Loading documents…", expanded=False) as status:
            pdf_docs, w1 = load_pdfs(files) if files else ([], [])
            progress.progress(15, text="Loading web pages…")
            web_docs, w2 = load_urls(urls) if urls else ([], [])
            all_warnings += w1 + w2
            docs = pdf_docs + web_docs
            status.update(label=f"Loaded {len(docs)} page(s)", state="complete")
        progress.progress(30, text="Chunking…")

        if not docs:
            progress.empty()
            for w in all_warnings:
                st.sidebar.warning(w)
            st.sidebar.error("No usable content was found. Nothing was indexed.")
            return

        # 2) Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
        if not chunks:
            progress.empty()
            st.sidebar.error("Documents produced no text chunks.")
            return
        progress.progress(40, text=f"Embedding {len(chunks)} chunks…")

        # 3) Embedding + storage (batched so we can show progress / respect rate limits)
        collection = f"rag_{uuid.uuid4().hex[:12]}"
        vs = Chroma(collection_name=collection, embedding_function=get_embeddings())  # in-memory
        total = len(chunks)
        for i in range(0, total, EMBED_BATCH_SIZE):
            vs.add_documents(chunks[i : i + EMBED_BATCH_SIZE])
            pct = 40 + int(60 * min(i + EMBED_BATCH_SIZE, total) / total)
            progress.progress(pct, text=f"Embedding… {min(i + EMBED_BATCH_SIZE, total)}/{total}")

        st.session_state.vectorstore = vs
        st.session_state.collection_name = collection
        st.session_state.stats = {
            "files": len({d.metadata["source"] for d in pdf_docs}),
            "urls": len({d.metadata["source"] for d in web_docs}),
            "chunks": total,
        }
        st.session_state.messages = []  # new knowledge base -> fresh chat
        progress.progress(100, text="Ready")

    except Exception as e:
        drop_vectorstore()
        progress.empty()
        st.sidebar.error(f"Indexing failed: {e}")
    finally:
        for w in all_warnings:
            st.sidebar.warning(w)


# --------------------------------------------------------------------------- #
# Chat helpers
# --------------------------------------------------------------------------- #
def render_sources(sources: List[dict]) -> None:
    with st.expander("View Source Context"):
        for i, s in enumerate(sources, 1):
            meta = s["metadata"]
            label = meta.get("source", "unknown")
            page = f" · page {meta['page'] + 1}" if isinstance(meta.get("page"), int) else ""
            st.markdown(f"**{i}. {label}{page}**")
            st.code(s["content"], language=None, wrap_lines=True)
            if meta.get("title"):
                st.caption(f"Title: {meta['title']}")


def answer_stream(llm: ChatMistralAI, context: str, question: str):
    chain = PROMPT | llm
    for chunk in chain.stream({"context": context, "question": question}):
        if chunk.content:
            yield chunk.content


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def sidebar() -> str:
    st.sidebar.title(" Knowledge Base")

    st.sidebar.subheader("1 · Upload PDFs")
    files = st.sidebar.file_uploader(
        "PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdfs_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )

    st.sidebar.subheader("2 · Add web URLs")
    raw_urls = st.sidebar.text_area(
        "One URL per line",
        placeholder="https://example.com/article\nhttps://another.site/page",
        key=f"urls_{st.session_state.urls_key}",
        height=100,
    )

    if st.sidebar.button("Process & Index", type="primary", use_container_width=True):
        urls = parse_urls(raw_urls)
        if not files and not urls:
            st.sidebar.warning("Add at least one PDF or URL first.")
        else:
            build_index(files, urls)

    if st.session_state.stats:
        s = st.session_state.stats
        st.sidebar.markdown(
            f'<span class="stat-pill">📄 {s["files"]} PDF(s)</span>'
            f'<span class="stat-pill">🌐 {s["urls"]} URL(s)</span>'
            f'<span class="stat-pill">🧩 {s["chunks"]} chunks</span>',
            unsafe_allow_html=True,
        )

    st.sidebar.divider()
    model_name = st.sidebar.text_input("Chat model", value=DEFAULT_CHAT_MODEL,
                                       help="e.g. labs-leanstral-1-5 or mistral-small-latest")
    if st.sidebar.button("🗑️ Clear Session", use_container_width=True):
        clear_session()
        st.rerun()
    return model_name


def main() -> None:
    init_state()

    if not os.getenv("MISTRAL_API_KEY"):
        st.error("`MISTRAL_API_KEY` is missing. Add it to your `.env` file "
                 "(`MISTRAL_API_KEY=your_key`) and restart the app.")
        st.stop()

    model_name = sidebar()

    st.title("RAG Assistant")
    st.caption("Ask questions grounded strictly in your PDFs and web pages.")

    # Replay history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                render_sources(m["sources"])

    ready = st.session_state.vectorstore is not None
    if not ready:
        st.info("Upload PDFs and/or add URLs in the sidebar, then click **Process & Index**.")

    query = st.chat_input("Ask a question about your documents…", disabled=not ready)
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            retriever = st.session_state.vectorstore.as_retriever(
                search_type="mmr", search_kwargs=RETRIEVER_KWARGS
            )
            with st.spinner("Retrieving context…"):
                retrieved = retriever.invoke(query)
            context = "\n\n".join(
                f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in retrieved
            )

            llm = get_llm(model_name.strip() or DEFAULT_CHAT_MODEL)
            answer = st.write_stream(answer_stream(llm, context, query))

            sources = [{"content": d.page_content, "metadata": d.metadata} for d in retrieved]
            if sources:
                render_sources(sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            msg = f"Something went wrong while answering: `{e}`"
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg, "sources": []})


if __name__ == "__main__":
    main()