import os
import shutil
import streamlit as st
import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama

from reranker import rerank


# ---------------- UI ----------------
st.set_page_config(page_title="RAG Document System", layout="wide")
st.title("📚 Intelligent Document Q&A System")


# ---------------- Embeddings ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": False}
    )


# ---------------- LLM (OLLAMA) ----------------
@st.cache_resource
def load_llm():
    return ChatOllama(
        model="llama3.1",
        temperature=0.2
    )


# ---------------- Reset Vector Store ----------------
def reset_vectorstore():
    for index_name in ["faiss_index_custom", "faiss_index"]:
        if os.path.isdir(index_name):
            shutil.rmtree(index_name)

    st.session_state.vectorstore = None


# ---------------- Process CSV ----------------
def process_document(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        original_len = len(df)

        sample_count = min(1000, max(1, original_len // 2))

        if sample_count < original_len:
            df = df.head(sample_count).reset_index(drop=True)
            st.info(f"Using first {sample_count} rows out of {original_len}")
        else:
            st.info(f"Using all {original_len} rows")

        st.write("Detected CSV columns:", list(df.columns))

        rows = df.to_dict(orient="records")

        def format_row_text(row):
            question = row.get("question_text") or row.get("question") or row.get("query") or ""
            long_answer = row.get("long_answer") or row.get("long_answers") or row.get("answer") or row.get("answer_text") or ""
            short_answer = row.get("short_answers") or ""
            context = row.get("context") or row.get("document") or ""

            parts = []
            if question:
                parts.append(f"Question: {question}")
            if long_answer:
                parts.append(f"Answer: {long_answer}")
            if short_answer:
                if isinstance(short_answer, list):
                    parts.append("Short Answers: " + ", ".join(map(str, short_answer)))
                else:
                    parts.append(f"Short Answers: {short_answer}")
            if context:
                parts.append(f"Context: {context}")

            if not parts:
                return " | ".join(str(v) for v in row.values() if pd.notna(v))

            return "\n".join(parts)

        texts = [format_row_text(r) for r in rows if format_row_text(r).strip()]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        docs = []
        for idx, text in enumerate(texts, start=1):
            chunks = splitter.create_documents([text])
            for doc in chunks:
                doc.metadata["row"] = idx
                docs.append(doc)

        st.info(f"Created {len(docs)} chunks from {len(texts)} rows")
        return docs

    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")
        return None


# ---------------- Session Init ----------------
if "loaded" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.loaded = True


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("📤 Upload CSV")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    upload_mode = st.radio(
        "Upload action:",
        ["Create new index", "Append to existing index"]
    )

    if uploaded_file:
        try:
            preview_df = pd.read_csv(uploaded_file, nrows=3)
            st.subheader("📋 Preview")
            st.dataframe(preview_df, use_container_width=True)
            uploaded_file.seek(0)
        except Exception as e:
            st.error(f"Preview error: {str(e)}")

        if st.button("🚀 Process & Add"):
            docs = process_document(uploaded_file)

            if docs:
                embeddings = load_embeddings()

                with st.spinner("Creating embeddings..."):
                    try:
                        vectorstore = st.session_state.vectorstore

                        if upload_mode == "Create new index" or vectorstore is None:
                            vectorstore = FAISS.from_documents(docs, embeddings)
                        else:
                            vectorstore.add_documents(docs)

                        vectorstore.save_local("faiss_index_custom")
                        st.session_state.vectorstore = vectorstore

                        st.success("Index updated successfully")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Vector error: {str(e)}")

    if st.button("🧹 Reset Index"):
        reset_vectorstore()
        st.success("Reset complete")
        st.rerun()

    st.subheader("📊 Vector Store Info")
    vs = st.session_state.vectorstore
    total = vs.index.ntotal if vs else 0
    st.metric("Total Vectors", total)


# ---------------- Query UI ----------------
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input("❓ Ask a question about your data:")

with col2:
    search_button = st.button("🔍 Search", type="primary")


# ---------------- RAG PIPELINE ----------------
if (query and search_button) or (query and st.session_state.get("last_query") != query):

    st.session_state.last_query = query
    vectorstore = st.session_state.vectorstore

    if vectorstore is None:
        st.warning("Please upload a CSV first")
    else:
        with st.spinner("Searching..."):
            try:
                results = vectorstore.similarity_search(query, k=5)

                if not results:
                    st.info("No results found")
                else:
                    doc_texts = [d.page_content for d in results]

                    # rerank
                    ranked = rerank(query, doc_texts)
                    top_chunks = ranked[:3]

                    # ---------------- LLM ----------------
                    llm = load_llm()

                    context = "\n\n".join(top_chunks)

                    prompt = f"""
                                You are a helpful assistant.

                                Use ONLY the context below to answer the question.

                                Context:
                                {context}

                                Question:
                                {query}

                                Rules:
                                - If answer is not in context, say "I don't know based on the data"
                                - Keep answer concise
                            """

                    response = llm.invoke(prompt)

                    st.subheader("🧠 AI Answer (Llama3.1)")
                    st.write(response.content)

                    st.subheader("📄 Sources")

                    for i, chunk in enumerate(top_chunks, 1):
                        with st.expander(f"Source {i}", expanded=(i == 1)):
                            st.write(chunk)
                            st.caption(f"{len(chunk)} chars")

            except Exception as e:
                st.error(f"Search error: {str(e)}")