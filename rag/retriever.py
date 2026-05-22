import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

embedder = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PATH)


def retrieve(query: str, candidate_id: str, top_k=3) -> list[str]:
    """Return top_k most relevant chunks for a query."""
    collection = client.get_or_create_collection(name=candidate_id)
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results["documents"][0]  # list of strings
