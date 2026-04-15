"""
Fast ingestion test with a small subset of the CSV for quick testing.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import csv


def build_vectorstore_sample():
    
    texts = []
    with open("myenv/data/Natural-Questions-Filtered.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 1000:  # Limit to 1000 rows
                break
            question = row.get('question', '').strip()
            long_answer = row.get('long_answers', '').strip()
            text = f"{question} {long_answer}" if long_answer else question
            if text.strip():
                texts.append(text)
    
    print(f"✅ Loaded {len(texts)} documents from CSV (sample)")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    
    docs = splitter.create_documents(texts)
    print(f"✅ Created {len(docs)} documents from splitter")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": False}
    )
    
    print("⏳ Generating embeddings (this may take a few minutes)...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Save with a different name to distinguish from full build
    vectorstore.save_local("faiss_index_sample")
    
    print("✅ Vector store sample created at faiss_index_sample/")
    print(f"   Total vectors: {len(docs)}")

if __name__ == "__main__":
    build_vectorstore_sample()
