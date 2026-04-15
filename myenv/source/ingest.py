from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from loader import load_nq_data

DATA_PATH = "myenv/data/Natural-Questions-Filtered.csv"

def build_vectorstore():
    raw_texts = load_nq_data(DATA_PATH)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = splitter.create_documents(raw_texts)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": False}
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local("faiss_index")

    print("✅ Vector store created!")

if __name__ == "__main__":
    build_vectorstore()
