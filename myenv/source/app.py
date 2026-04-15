import os
import shutil
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from reranker import rerank
import pandas as pd

st.set_page_config(page_title="RAG Document System", layout="wide")
st.title("📚 Intelligent Document Q&A System")

# --- Load embeddings ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": False}
    )

# --- Load vector store ---

def reset_vectorstore():
    for index_name in ["faiss_index_custom", "faiss_index"]:
        if os.path.isdir(index_name):
            shutil.rmtree(index_name)
    st.session_state.vectorstore = None

# --- Process CSV file ---
def process_document(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        original_len = len(df)

        # Use only 50% of the CSV rows, capped at 1000 records.
        # Use a deterministic slice so the same rows are loaded every time.
        sample_count = min(1000, max(1, original_len // 2))
        if sample_count < original_len:
            df = df.head(sample_count).reset_index(drop=True)
            st.info(f"Using first {sample_count} rows out of {original_len} rows (50% sample capped at 1000)")
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
                return " | ".join(str(value) for value in row.values() if pd.notna(value) and str(value).strip())
            return "\n".join(parts)

        texts = []
        for row in rows:
            text = format_row_text(row)
            if text.strip():
                texts.append(text)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        docs = []
        for idx, text in enumerate(texts, start=1):
            row_docs = splitter.create_documents([text])
            for doc in row_docs:
                doc.metadata["row"] = idx
                docs.append(doc)

        st.info(f"Created {len(docs)} document chunks from {len(texts)} rows")
        return docs

    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")
        return None


if "loaded" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.loaded = True

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("📤 Upload CSV")
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )

    upload_mode = st.radio("Upload action:", ["Create new index", "Append to existing index"])
    st.caption("Create new index to start from zero, or append to add data to existing vectors.")

    if uploaded_file:
        # Preview first 3 records
        try:
            preview_df = pd.read_csv(uploaded_file, nrows=3)
            st.subheader("📋 Data Preview (First 3 Rows)")
            st.dataframe(preview_df, use_container_width=True)
            # Reset file pointer for processing
            uploaded_file.seek(0)
        except Exception as e:
            st.error(f"Error reading CSV for preview: {str(e)}")

        if st.button("🚀 Process & Add to Vector Store"):
            docs = process_document(uploaded_file)

            if docs:
                embeddings = load_embeddings()

                with st.spinner(f"Processing {len(docs)} chunks..."):
                    try:
                        vectorstore = st.session_state.vectorstore
                        if upload_mode == "Create new index" or vectorstore is None:
                            vectorstore = FAISS.from_documents(docs, embeddings)
                        else:
                            vectorstore.add_documents(docs)

                        vectorstore.save_local("faiss_index_custom")
                        st.session_state.vectorstore = vectorstore
                        st.success(f"✅ {upload_mode} completed: {len(docs)} chunks added")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error adding to vector store: {str(e)}")

    if st.button("🧹 Reset saved vector store"):
        reset_vectorstore()
        st.success("✅ Vector store reset. Reload the app to start fresh.")
        st.rerun()

    # Store info
    st.subheader("📊 Vector Store Info")
    vectorstore = st.session_state.vectorstore
    total_vectors = vectorstore.index.ntotal if vectorstore else 0
    st.metric("Total Vectors", total_vectors)
    if not vectorstore:
        st.info("No vector store loaded. Upload a CSV file to create vectors.")

# --- Query interface ---
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input("❓ Ask a question about your data:")

with col2:
    search_button = st.button("🔍 Search", type="primary")

# --- Results ---
if (query and search_button) or (query and st.session_state.get("last_query") != query):
    st.session_state.last_query = query
    vectorstore = st.session_state.vectorstore

    if vectorstore is None:
        st.warning("⚠️ Please upload a CSV file first.")
    else:
        with st.spinner("🔍 Searching..."):
            try:
                results = vectorstore.similarity_search(query, k=5)

                if not results:
                    st.info("No relevant results found.")
                else:
                    doc_texts = [doc.page_content for doc in results]

                    # rerank always ON (since settings removed)
                    ranked_texts = rerank(query, doc_texts)
                    final_results = ranked_texts[:3]

                    st.subheader(f"📄 Found {len(final_results)} Results")

                    for idx, text in enumerate(final_results, 1):
                        with st.expander(f"📖 Result {idx}", expanded=(idx == 1)):
                            st.write(text)
                            st.caption(f"Length: {len(text)} characters")

                    

            except Exception as e:
                st.error(f"Search error: {str(e)}")

