import chromadb
import ollama
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.auto import partition
import os

CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
SUMMARY_LLM = "qwen3"

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


def get_summary_collection(candidate_id: str):
    return client.get_or_create_collection(name=f"{candidate_id}_summaries")


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


# -- Summary generation ------------------------------------------------------

def generate_summary(full_text: str, doc_type: str) -> str:
    """Ask the LLM to summarize the document in 5-6 sentences.

    Used to populate the summary index — searched when queries are broad
    like 'tell me about this candidate' rather than specific skill lookups.
    """
    prompt = (
        f"You are summarizing a {doc_type} document for a recruiter.\n"
        f"Write a concise 5-6 sentence summary covering the most important points.\n"
        f"Be factual, no opinions.\n\n"
        f"Document:\n{full_text[:3000]}"  # cap at 3000 chars to avoid huge prompts
    )

    response = ollama.chat(
        model=SUMMARY_LLM,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


# -- Ingestion pipeline -------------------------------------------------------

def ingest_document(file_path: str, candidate_id: str, doc_type: str = "cv"):
    """Full pipeline: file -> sections -> chunks -> embeddings -> ChromaDB.
    Also generates and stores a document summary in a separate summary index.

    Each chunk is stored with metadata:
        - candidate_id: who this document belongs to
        - doc_type: cv | readme | certificate | recommendation
        - source_file: original filename
        - section: which section of the document this chunk came from

    Summary index stores one summary per document for broad queries.
    """
    sections = extract_sections(file_path)
    if not sections:
        print(f"Warning: No content extracted from {file_path}")
        return

    collection = get_collection(candidate_id)
    base = os.path.basename(file_path)

    # --- Chunk index ---
    all_chunks, all_embeddings, all_ids, all_metas = [], [], [], []

    for s_idx, section in enumerate(sections):
        # Split the body text
        chunks = text_splitter.split_text(section["text"])

        for c_idx, chunk in enumerate(chunks):
            # Prepend the section title to the text chunk for semantic richness
            contextualized_chunk = f"Section: {section['section']}\n{chunk}"

            all_chunks.append(contextualized_chunk)
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

    # --- Summary index ---
    full_text = " ".join(s["text"] for s in sections)
    summary = generate_summary(full_text, doc_type)

    summary_collection = get_summary_collection(candidate_id)
    summary_embedding = embedder.encode([summary]).tolist()
    summary_collection.add(
        documents=[summary],
        embeddings=summary_embedding,
        ids=[base],
        metadatas=[{
            "candidate_id": candidate_id,
            "doc_type": doc_type,
            "source_file": base,
        }],
    )
    print(f"Summary stored for {base}: '{summary[:80]}...'")