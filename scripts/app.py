import streamlit as st
from chatbot import generate_answer
import os
from file_loader import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
from file_loader import chunk_text
from embeddings import create_vector_store, load_vector_store
st.title("🔍 AI Personal Knowledge Chatbot")

uploaded_file = st.file_uploader("Upload your notes (PDF, DOCX, TXT, MD)", type=["pdf", "docx", "txt", "md"])

if uploaded_file:
    file_path = os.path.abspath(uploaded_file.name)
    st.success("File uploaded successfully!")
    
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    elif file_path.endswith(".txt"):
        text = extract_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file format")
    text_chunks = chunk_text(text)
    create_vector_store(text_chunks)
    
vectors = load_vector_store()
question = st.text_input("Ask something from your documents:")
if st.button("Get Answer") and question:
    with st.spinner("Generating answer..."):
        answer = generate_answer(question)
        st.success(answer)
