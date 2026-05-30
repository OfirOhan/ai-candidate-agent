import chromadb
import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder

CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EXPANSION_LLM = "qwen3"

embedder = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder(RERANK_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PATH)


# ---------------------------------------------------------------------------
# 1. Query Expansion — use Ollama to generate search variations
# ---------------------------------------------------------------------------

def expand_query(original_query: str, n_variations: int = 3) -> list[str]:
    """Ask the LLM to rewrite the query into multiple search-friendly variations.

    Returns a list that always starts with the original query, followed by
    up to `n_variations` alternative phrasings.
    """
    prompt = (
        f"You are a search-query rewriter. Given the user question below, "
        f"generate exactly {n_variations} alternative search queries that "
        f"capture the same intent but use different keywords or phrasing.\n\n"
        f"User question: \"{original_query}\"\n\n"
        f"Return ONLY the queries, one per line, no numbering, no explanation."
    )

    response = ollama.chat(
        model=EXPANSION_LLM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response["message"]["content"].strip()

    # Parse the LLM output — one query per line, skip blanks
    variations = [line.strip() for line in raw.splitlines() if line.strip()]
    return [original_query] + variations[:n_variations]


# ---------------------------------------------------------------------------
# 2. Re-ranking — score every candidate chunk with a Cross-Encoder
# ---------------------------------------------------------------------------

def rerank(query: str, chunks: list[str], top_k: int = 3) -> list[str]:
    """Re-rank `chunks` by relevance to `query` using a Cross-Encoder.

    Returns the top_k most relevant chunks in descending relevance order.
    """
    if not chunks:
        return []

    pairs = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)

    # Pair each chunk with its score, sort descending, take top_k
    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


# ---------------------------------------------------------------------------
# 3. Main retrieve pipeline: expand → fetch → deduplicate → re-rank
# ---------------------------------------------------------------------------

def retrieve(query: str, candidate_id: str, top_k: int = 3) -> list[str]:
    """Full advanced retrieval pipeline.

    Steps:
        1. Expand the query into multiple variations via Ollama.
        2. For each variation, fetch candidate chunks from ChromaDB.
        3. Deduplicate the combined results.
        4. Re-rank all candidates with a Cross-Encoder.
        5. Return the top_k best chunks.
    """
    collection = client.get_or_create_collection(name=candidate_id)

    # --- Step 1: Query Expansion ---
    queries = expand_query(query)
    print(f"[Retriever] Expanded '{query}' → {queries}")

    # --- Step 2: Fetch chunks for every query variation ---
    fetch_per_query = 10  # cast a wide net per variation
    all_chunks: list[str] = []
    seen: set[str] = set()

    for q in queries:
        q_embedding = embedder.encode([q]).tolist()
        results = collection.query(
            query_embeddings=q_embedding,
            n_results=fetch_per_query,
        )
        for chunk in results["documents"][0]:
            if chunk not in seen:
                seen.add(chunk)
                all_chunks.append(chunk)

    print(f"[Retriever] Fetched {len(all_chunks)} unique chunks across {len(queries)} queries")

    # --- Step 3: Re-rank and return the best ---
    best = rerank(query, all_chunks, top_k=top_k)
    print(f"[Retriever] Re-ranked → returning top {len(best)} chunks")
    return best
