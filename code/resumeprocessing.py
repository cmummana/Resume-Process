import sqlite3
import faiss
import numpy as np
from datetime import datetime
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.partition.auto import partition
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
import tensorflow as tf
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')
import types
import torch

# Workaround to prevent torch.classes error in Streamlit
if not hasattr(torch, "classes"):
    torch.classes = types.SimpleNamespace()
else:
    if not isinstance(torch.classes, types.SimpleNamespace):
        torch.classes.__path__ = []


# Paths
DOCUMENT_FOLDER = r"C:\Users\Chaitrika\Desktop\Documents"

os.makedirs(DOCUMENT_FOLDER, exist_ok=True)

# Load model
model = SentenceTransformer("BAAI/bge-base-en")

# DB setup
def init_db():
    conn = sqlite3.connect("documents.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id INTEGER PRIMARY KEY, filename TEXT, source TEXT, timestamp TEXT, content TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS chunks
                 (chunk_id INTEGER PRIMARY KEY, document_id INTEGER, chunk_text TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS embeddings
                 (chunk_id INTEGER PRIMARY KEY, embedding BLOB)''')
    conn.commit()
    return conn

# Text extraction and chunking
def extract_text_and_metadata(filepath):
    elements = partition(filepath)
    full_text = "\n".join([el.text for el in elements if hasattr(el, "text")])
    metadata = {
        "filename": os.path.basename(filepath),
        "source": filepath,
        "timestamp": datetime.now().isoformat(),
        "text": full_text
    }
    return metadata

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def generate_embeddings(chunks):
    return model.encode(chunks, show_progress_bar=True)

def save_to_db(conn, metadata, chunks, embeddings):
    c = conn.cursor()
    c.execute("SELECT id FROM documents WHERE filename = ?", (metadata["filename"],))
    if c.fetchone():
        return  # already exists
    c.execute("INSERT INTO documents (filename, source, timestamp, content) VALUES (?, ?, ?, ?)",
              (metadata["filename"], metadata["source"], metadata["timestamp"], metadata["text"]))
    document_id = c.lastrowid
    for chunk, embedding in zip(chunks, embeddings):
        c.execute("INSERT INTO chunks (document_id, chunk_text) VALUES (?, ?)", (document_id, chunk))
        chunk_id = c.lastrowid
        c.execute("INSERT INTO embeddings (chunk_id, embedding) VALUES (?, ?)", (chunk_id, embedding.tobytes()))
    conn.commit()

def load_chunks_and_embeddings():
    conn = sqlite3.connect("documents.db")
    c = conn.cursor()
    c.execute('''SELECT chunks.chunk_text, embeddings.embedding, chunks.document_id
                 FROM chunks JOIN embeddings ON chunks.chunk_id = embeddings.chunk_id''')
    results = c.fetchall()
    conn.close()

    chunks, embeddings, doc_ids = [], [], []
    for chunk_text, embedding_blob, doc_id in results:
        chunks.append(chunk_text)
        embeddings.append(np.frombuffer(embedding_blob, dtype=np.float32))
        doc_ids.append(doc_id)

    return chunks, embeddings, doc_ids

# Vector Index
class VectorIndex:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.chunk_map = []

    def add(self, embeddings, chunks, doc_ids):
        self.index.add(np.array(embeddings).astype("float32"))
        self.chunk_map.extend(zip(chunks, doc_ids))

    def search(self, query_embedding, top_k=5):
        D, I = self.index.search(np.array(query_embedding).astype("float32"), k=top_k)
        return [(self.chunk_map[i], D[0][idx]) for idx, i in enumerate(I[0])]

# ============== Streamlit UI ==============
st.set_page_config(page_title="📄 Resume Semantic Search", layout="wide")
st.title("📄 Resume Semantic Search Tool")

# Step 1: Process documents from folder
st.subheader("📁 Upload Resumes")

conn = init_db()

# Upload resumes from UI
uploaded_files = st.file_uploader(
    "Upload one or more resumes",
    type=["pdf", "docx", "doc", "txt"],
    accept_multiple_files=True
)


# Show a process button only if files are uploaded
if uploaded_files:
    if st.button("🚀 Process Uploaded Resumes"):
        with st.spinner("Processing uploaded resumes..."):
            c = conn.cursor()
            processed = 0
            skipped = 0

            for uploaded_file in uploaded_files:
                filename = uploaded_file.name
                filepath = os.path.join(DOCUMENT_FOLDER, filename)

                # Save file to disk
                with open(filepath, "wb") as f:
                    f.write(uploaded_file.read())

                # Check if already processed
                c.execute("SELECT 1 FROM documents WHERE filename = ?", (filename,))
                if c.fetchone():
                    skipped += 1
                    continue

                metadata = extract_text_and_metadata(filepath)
                chunks = chunk_text(metadata["text"])
                embeddings = generate_embeddings(chunks)
                save_to_db(conn, metadata, chunks, embeddings)
                processed += 1

            st.success(f"✅ {processed} resume(s) processed. ⏭️ {skipped} skipped (already uploaded).")

# Step 2: Search
st.subheader("Semantic Search")

# Load all chunks and vector index
chunks, embeddings, doc_ids = load_chunks_and_embeddings()
vector_index = None
if chunks:
    vector_index = VectorIndex(dim=len(embeddings[0]))
    vector_index.add(embeddings, chunks, doc_ids)

# Suggested search examples
suggested_terms = [
    "Python developer with 3 years experience",
    "Java developer in Bangalore",
    "Data engineer with AWS",
    "Frontend developer with React",
    "Machine learning and NLP experience"
]

query = st.text_input(" Enter your search query")

# Show clickable suggestions
st.markdown("** Suggested Queries:**")
for term in suggested_terms:
    if st.button(term):
        query = term

if query and vector_index:
    query_vec = model.encode([query])
    results = vector_index.search(query_vec, top_k=10)

    c = conn.cursor()
    seen = set()
    unique_results = []

    for (chunk_text, doc_id), distance in results:
        c.execute("SELECT filename FROM documents WHERE id = ?", (doc_id,))
        filename_row = c.fetchone()
        filename = filename_row[0] if filename_row else "Unknown"
        if filename not in seen:
            similarity = round(1 - distance, 2)  # Convert L2 distance to similarity
            unique_results.append((query, filename, chunk_text, similarity))
            seen.add(filename)

    # Build and show DataFrame
    result_df = pd.DataFrame(unique_results, columns=[
        "Search Query", "Filename", "Matched Text Snippet", "Similarity Score"
    ])

    st.markdown("Search Results")
    st.dataframe(result_df, use_container_width=True)

elif query:
    st.warning("No documents indexed. Please click 'Start Processing' first.")
