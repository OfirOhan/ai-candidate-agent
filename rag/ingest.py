import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.auto import partition
import os

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "../chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"

embedder = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PATH)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


def get_collection(candidate_id: str):
    return client.get_or_create_collection(name=candidate_id)


# -- Section extraction ------------------------------------------------------

def extract_sections(file_path: str) -> list[dict]:
    """Partition the document into sections using Unstructured.

    Returns a list of {text, section} dicts.
    Works for PDF, DOCX, MD, images, TXT — any language.
    """
    elements = partition(file_path)

    sections = []
    current_section = "general"

    for element in elements:
        if element.category == "Title":
            current_section = element.text.strip()
        elif element.text.strip():
            sections.append({
                "text": element.text.strip(),
                "section": current_section,
            })

    return sections


# -- Ingestion pipeline -------------------------------------------------------

def ingest_document(file_path: str, candidate_id: str, doc_type: str = "cv"):
    """Full pipeline: file -> sections -> chunks -> embeddings -> ChromaDB.

    Each chunk is stored with metadata:
        - candidate_id: who this document belongs to
        - doc_type: cv | readme | certificate | recommendation
        - source_file: original filename
        - section: which section of the document this chunk came from
    """
    sections = extract_sections(file_path)
    if not sections:
        print(f"Warning: No content extracted from {file_path}")
        return

    collection = get_collection(candidate_id)
    base = os.path.basename(file_path)

    all_chunks, all_embeddings, all_ids, all_metas = [], [], [], []

    for s_idx, section in enumerate(sections):
        # Chunk within each section separately — sections never bleed into each other
        chunks = text_splitter.split_text(section["text"])
        for c_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{base}_s{s_idx}_chunk_{c_idx}")
            all_metas.append({
                "candidate_id": candidate_id,
                "doc_type": doc_type,
                "source_file": base,
                "section": section["section"],
            })

    all_embeddings = embedder.encode(all_chunks).tolist()

    collection.add(
        documents=all_chunks,
        embeddings=all_embeddings,
        ids=all_ids,
        metadatas=all_metas,
    )
    print(f"Ingested {len(all_chunks)} chunks from {base} ({len(sections)} sections)")