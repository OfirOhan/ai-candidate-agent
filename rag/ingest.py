import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
import os

CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

embedder = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection(candidate_id: str):
    return client.get_or_create_collection(name=candidate_id)


def chunk_text(text: str, chunk_size=300, overlap=50) -> list[str]:
    """Split text into overlapping chunks of ~chunk_size characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    return " ".join(page.get_text() for page in doc)


def ingest_document(file_path: str, candidate_id: str):
    """Full pipeline: PDF -> text -> chunks -> embeddings -> ChromaDB."""
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks).tolist()
    collection = get_collection(candidate_id)

    ids = [f"{os.path.basename(file_path)}_chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    print(f"Ingested {len(chunks)} chunks from {file_path}")
