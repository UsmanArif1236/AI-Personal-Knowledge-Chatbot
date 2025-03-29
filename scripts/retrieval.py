from embeddings import load_vector_store

def retrieve_relevant_chunks(question):
    """Retrieves the most relevant document chunks from FAISS."""
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    
    return retriever.get_relevant_documents(question)
