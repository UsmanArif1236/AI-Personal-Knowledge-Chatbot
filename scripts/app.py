import streamlit as st
from chatbot import generate_answer
import os
from file_loader import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
from file_loader import chunk_text
from embeddings import create_vector_store, load_vector_store

st.set_page_config(page_title="AI Personal Knowledge Chatbot", layout="wide")

st.sidebar.title("📝 Chat History")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.sidebar.button("🆕 New Chat"):
    st.session_state.chat_history = []

st.title("🔍 AI Personal Knowledge Chatbot")

uploaded_file = st.file_uploader("Upload your notes (PDF, DOCX, TXT, MD)", type=["pdf", "docx", "txt", "md"])
if uploaded_file:
    file_path = os.path.abspath(uploaded_file.name)
    save_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(save_path)
    elif file_path.endswith(".docx"):
        text = extract_text_from_docx(save_path)
    elif file_path.endswith(".txt"):
        text = extract_text_from_txt(save_path)
    else:
        raise ValueError("Unsupported file format")
    
    text_chunks = chunk_text(text)
    create_vector_store(text_chunks)
    st.success("File uploaded and processed successfully!")
    
vectors = load_vector_store()
question = st.text_input("Ask something from your documents:")

if st.button("Get Answer") and question:
    with st.spinner("Generating answer..."):
        answer = generate_answer(question)
        st.session_state.chat_history.append((question, answer))

st.subheader("💬 Chat")
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
