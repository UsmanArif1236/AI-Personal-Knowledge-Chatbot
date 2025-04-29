import faiss
import pickle

FAISS_INDEX_FILE = "index.faiss"
TEXT_CHUNKS_FILE = "index.pkl"

index = faiss.read_index(FAISS_INDEX_FILE)
print("FAISS Index Loaded!")
print("Number of Vectors:", index.ntotal)

with open(TEXT_CHUNKS_FILE, "rb") as f:
    text_chunks = pickle.load(f)
print("Stored Text Chunks Loaded!")

for i in range(min(5, len(text_chunks))):
    print(f"Index {i}: {text_chunks[i]}")
