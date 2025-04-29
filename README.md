# AI Personal Knowledge Chatbot 

AI-powered chatbot that allows users to upload personal notes (PDF, DOCX, TXT, MD) and intelligently query information using LLMs and vector search.

---

## Project Description

Finding specific information inside long documents can be tedious and inefficient.  
This project solves that by building a chatbot that retrieves the most relevant parts of your uploaded notes and generates accurate, context-aware answers using a powerful LLM.

---

## Features

- 📄 Upload PDF, DOCX, TXT, or MD files
- 🧩 Automatic text chunking using LangChain
- 🧠 Semantic search using FAISS vector store
- 💬 Smart answers via LLaMA-3 70B model (through Groq API)
- 🔁 Maintains conversation context
- 🌐 Streamlit-based user interface

---

## Tech Stack

| Technology        | Usage                          |
|-------------------|--------------------------------|
| **Python**        | Core Programming Language      |
| **Streamlit**     | Web Interface                  |
| **FAISS**         | Vector Store                   |
| **LangChain**     | Chunking, Retrieval            |
| **GROQ / LLaMA-3**| Large Language Model (via API) |
| **HuggingFace**   | Embedding Model                |
| **PyPDF2/docx**   | Document Parsing               |

---

## How It Works

1. User uploads a document (PDF, DOCX, TXT, MD)
2. The document is parsed and split into chunks
3. Chunks are embedded using `sentence-transformers/all-mpnet-base-v2`
4. Embeddings are stored in a FAISS vector database
5. On asking a question:
    - Relevant chunks are retrieved
    - LLaMA-3 70B generates a detailed, context-aware answer
6. Chat history is preserved across messages

