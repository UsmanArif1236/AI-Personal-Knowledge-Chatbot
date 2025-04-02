from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from pathlib import Path


def create_vector_store(text_chunks):
    """Converts text into embeddings and stores in FAISS."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    parent_dir = Path.cwd().parent  # Get parent directory
    VECTOR_DIR = parent_dir / "vectorstore/"
    os.makedirs(VECTOR_DIR, exist_ok=True)
    vector_store.save_local(VECTOR_DIR)
    
    return vector_store

def load_vector_store():
    """Loads FAISS vector database."""
    parent_dir = Path.cwd().parent  # Get parent directory
    VECTOR_DIR = parent_dir / "vectorstore/"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.load_local(VECTOR_DIR, embeddings,allow_dangerous_deserialization=True)
