import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

embedder = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Recursive splitter: tries to split on paragraphs first ("\n\n"),
# then sentences ("\n", ". "), then words (" "), preserving meaning.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", ", ", " ", ""],
    length_function=len,
)


def get_collection(candidate_id: str):
    return client.get_or_create_collection(name=candidate_id)


def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    return " ".join(page.get_text() for page in doc)


def ingest_document(file_path: str, candidate_id: str):
    """Full pipeline: PDF -> text -> smart chunks -> embeddings -> ChromaDB."""
    text = extract_text_from_pdf(file_path)
    chunks = text_splitter.split_text(text)
    embeddings = embedder.encode(chunks).tolist()
    collection = get_collection(candidate_id)

    ids = [f"{os.path.basename(file_path)}_chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    print(f"Ingested {len(chunks)} chunks from {file_path}")
