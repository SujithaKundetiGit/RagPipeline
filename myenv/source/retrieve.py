"""
Retrieval script to query documents from the FAISS vector store using natural language.
"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_vectorstore(index_name="faiss_index_sample"):
    """Load the FAISS vector store from disk."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": False}
    )
    vectorstore = FAISS.load_local(
        index_name, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    return vectorstore

def retrieve_documents(query, k=5, index_name="faiss_index_sample"):
    """Retrieve top K documents most similar to the query."""
    vectorstore = load_vectorstore(index_name)
    results = vectorstore.similarity_search(query, k=k)
    
    return results

def main():
    print("🔍 Document Retrieval System")
    print("=" * 60)
    
    # Example queries
    queries = [
        "What is machine learning?",
        "How does AI work?",
        "Tell me about neural networks"
    ]
    
    for query in queries:
        print(f"\n📌 Query: '{query}'")
        print("-" * 60)
        
        try:
            results = retrieve_documents(query, k=3)
            
            for i, doc in enumerate(results, 1):
                text = doc.page_content
                # Truncate for display
                display_text = text[:200] + "..." if len(text) > 200 else text
                print(f"{i}. {display_text}")
                print()
        except Exception as e:
            print(f"❌ Error retrieving documents: {str(e)}")
            print("Make sure ingest.py or ingest_sample.py has completed successfully!")

if __name__ == "__main__":
    main()
