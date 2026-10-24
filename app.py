import os
import gc

import streamlit as st

from dotenv import load_dotenv

from utils.pdf_processor import (
    extract_pdf_pages,
    create_chunks
)

from utils.embeddings import EmbeddingModel

from utils.vector_store import VectorStore

from utils.rag import RAGAssistant

from utils.summarizer import DocumentSummarizer

from utils.quiz_generator import QuizGenerator


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "documents" not in st.session_state:

    st.session_state.documents = []


if "chunks" not in st.session_state:

    st.session_state.chunks = []


if "vector_store" not in st.session_state:

    st.session_state.vector_store = None


if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# --------------------------------------------------
# INITIALIZE AI
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():

    return EmbeddingModel()


@st.cache_resource
def load_rag():

    return RAGAssistant()


@st.cache_resource
def load_summarizer():

    return DocumentSummarizer()


@st.cache_resource
def load_quiz_generator():

    return QuizGenerator()


embedding_model = load_embedding_model()

rag = load_rag()

summarizer = load_summarizer()

quiz_generator = load_quiz_generator()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📚 AI Document Intelligence Assistant")

st.markdown(
    """
Upload your documents and use AI to:

- 🤖 Ask questions
- 🔎 Search documents
- 📝 Generate summaries
- 🎓 Create quizzes
- 📌 Get source references
"""
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Controls")

    language = st.selectbox(
        "Response Language",
        [
            "English",
            "Hindi",
            "Gujarati"
        ]
    )

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("🗑️ Clear Documents"):

        st.session_state.documents = []

        st.session_state.chunks = []

        st.session_state.vector_store = None

        st.session_state.chat_history = []

        st.success("Documents cleared.")

        st.rerun()


# --------------------------------------------------
# PROCESS DOCUMENTS
# --------------------------------------------------

if uploaded_files:

    if st.session_state.vector_store is None:

        all_chunks = []

        documents = []

        progress = st.progress(0)

        for i, uploaded_file in enumerate(uploaded_files):

            pages = extract_pdf_pages(
                uploaded_file
            )

            chunks = create_chunks(pages)

            for chunk in chunks:

                chunk["source"] = uploaded_file.name

            all_chunks.extend(chunks)

            documents.append({
                "name": uploaded_file.name,
                "pages": len(pages),
                "chunks": len(chunks)
            })

            progress.progress(
                (i + 1) / len(uploaded_files)
            )

        # ------------------------------------------
        # EMBEDDINGS
        # ------------------------------------------

        texts = [
            chunk["text"]
            for chunk in all_chunks
        ]

        embeddings = embedding_model.encode(
            texts
        )

        # ------------------------------------------
        # VECTOR STORE
        # ------------------------------------------

        vector_store = VectorStore(
            dimension=embeddings.shape[1]
        )

        vector_store.add_documents(
            all_chunks,
            embeddings
        )

        st.session_state.vector_store = vector_store

        st.session_state.chunks = all_chunks

        st.session_state.documents = documents

        st.success(
            f"Processed {len(uploaded_files)} document(s) successfully!"
        )


# --------------------------------------------------
# DOCUMENT DASHBOARD
# --------------------------------------------------

if st.session_state.documents:

    st.subheader("📊 Document Dashboard")

    total_pages = sum(
        d["pages"]
        for d in st.session_state.documents
    )

    total_chunks = len(
        st.session_state.chunks
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Documents",
            len(st.session_state.documents)
        )

    with col2:

        st.metric(
            "Pages",
            total_pages
        )

    with col3:

        st.metric(
            "Text Chunks",
            total_chunks
        )


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "💬 Chat",
        "📝 Summary",
        "🎓 Study Mode",
        "📄 Documents"
    ]
)


# ==================================================
# CHAT
# ==================================================

with tab1:

    st.subheader("💬 Ask Your Documents")

    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

    question = st.chat_input(
        "Ask something about your documents..."
    )

    if question:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):

            st.write(question)

        # ------------------------------------------
        # CREATE QUERY EMBEDDING
        # ------------------------------------------

        query_embedding = embedding_model.encode(
            [question]
        )[0]

        # ------------------------------------------
        # SEARCH
        # ------------------------------------------

        results = (
            st.session_state.vector_store.search(
                query_embedding,
                top_k=5
            )
        )

        # ------------------------------------------
        # BUILD CONTEXT
        # ------------------------------------------

        context = ""

        for result in results:

            context += f"""
SOURCE: {result['source']}
PAGE: {result['page']}

{result['text']}

-------------------------
"""

        # ------------------------------------------
        # GENERATE ANSWER
        # ------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching documents..."
            ):

                answer = rag.generate_answer(
                    question,
                    context,
                    language
                )

                st.write(answer)

                # ----------------------------------
                # SOURCES
                # ----------------------------------

                st.markdown(
                    "### 📌 Sources"
                )

                for result in results:

                    st.caption(
                        f"📄 {result['source']} — "
                        f"Page {result['page']}"
                    )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })


# ==================================================
# SUMMARY
# ==================================================

with tab2:

    st.subheader("📝 Document Summary")

    if st.session_state.chunks:

        if st.button("✨ Generate Summary"):

            text = "\n\n".join(
                chunk["text"]
                for chunk in st.session_state.chunks
            )

            with st.spinner(
                "Analyzing document and generating summary..."
            ):

                summary = summarizer.summarize(
                    text,
                    language
                )

            st.markdown(summary)

    else:

        st.info(
            "Upload a document first."
        )

# ==================================================
# STUDY MODE
# ==================================================

with tab3:

    st.subheader("🎓 AI Study Mode")

    if st.session_state.chunks:

        number = st.slider(
            "Number of Questions",
            3,
            15,
            5
        )

        if st.button(
            "🧠 Generate Quiz"
        ):

            text = "\n\n".join(
                chunk["text"]
                for chunk in st.session_state.chunks
            )

            text = text[:30000]

            with st.spinner(
                "Creating questions..."
            ):

                quiz = quiz_generator.generate_quiz(
                    text,
                    number
                )

            st.markdown(quiz)

    else:

        st.info(
            "Upload documents to start studying."
        )


# ==================================================
# DOCUMENTS
# ==================================================

with tab4:

    st.subheader("📄 Uploaded Documents")

    if st.session_state.documents:

        for document in st.session_state.documents:

            st.markdown(
                f"""
                ### 📄 {document['name']}

                **Pages:** {document['pages']}  
                **Text chunks:** {document['chunks']}
                """
            )

            st.divider()

    else:

        st.info(
            "No documents uploaded."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI Document Intelligence Assistant • "
    "Built with Streamlit + Groq + RAG + FAISS"
)

gc.collect()