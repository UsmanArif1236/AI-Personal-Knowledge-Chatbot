import faiss
import pickle

# Define paths
FAISS_INDEX_FILE = r"C:\Users\uarif\OneDrive\Documents\Projects\AI-Personal-Knowledge-Chabtbot\vectorstore\index.faiss"
TEXT_CHUNKS_FILE = r"C:\Users\uarif\OneDrive\Documents\Projects\AI-Personal-Knowledge-Chabtbot\vectorstore\index.pkl"

# Load the FAISS index
index = faiss.read_index(FAISS_INDEX_FILE)
print("FAISS Index Loaded!")
print("Number of Vectors:", index.ntotal)

# Load the stored text chunks
with open(TEXT_CHUNKS_FILE, "rb") as f:
    text_chunks = pickle.load(f)
print("Stored Text Chunks Loaded!")


# Print the first 5 text entries
for i in range(min(5, len(text_chunks))):  
    print(f"Index {i}: {text_chunks[i]}")

vector_id = 2  # Change this to the index you want to view
print(f"Original Text at Index {vector_id}: {text_chunks[0]}")
