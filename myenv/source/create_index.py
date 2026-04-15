from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Load embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Your sample documents (replace later with your real data)
texts = [
    "FAISS is a vector database used for similarity search.",
    "LangChain helps build RAG applications easily.",
    "Streamlit is used to build web apps in Python.",
]

# 3. Create FAISS index
db = FAISS.from_texts(texts, embeddings)

# 4. Save it locally
db.save_local("faiss_index")

print("✅ FAISS index created and saved successfully!")