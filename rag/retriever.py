import chromadb
import ollama
import numpy as np
from rank_bm25 import BM25Okapi
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

    variations = [line.strip() for line in raw.splitlines() if line.strip()]
    return [original_query] + variations[:n_variations]


# ---------------------------------------------------------------------------
# 2. Query Routing — summary index vs vector index
# ---------------------------------------------------------------------------

def is_broad_query(query: str, candidate_id: str, threshold: float = 0.3) -> bool:
    """Decide if the query is broad (overview) or specific (targeted lookup).

    Compares the query against actual summaries vs actual section names in ChromaDB.
    If the query is more similar to summaries than to any specific section → broad.

    Language agnostic — no hardcoded phrases, works in Hebrew, English, anything.

    Examples:
        broad:    "ספר לי על המועמד", "give me an overview"
        specific: "what FastAPI projects did he build", "כישורי Python"
    """
    collection = client.get_or_create_collection(name=candidate_id)
    chunk_results = collection.get(include=["metadatas"])
    sections = list({
        m["section"]
        for m in chunk_results["metadatas"]
        if "section" in m
    })

    summary_collection = client.get_or_create_collection(f"{candidate_id}_summaries")
    summary_results = summary_collection.get(include=["embeddings"])

    if not summary_results["embeddings"] or not sections:
        return False

    query_embedding = embedder.encode([query])[0]

    # Score query against all section names
    section_embeddings = embedder.encode(sections)
    best_section_score = float(np.max(section_embeddings @ query_embedding))

    # Score query against actual summary embeddings
    summary_embeddings = np.array(summary_results["embeddings"])
    best_summary_score = float(np.max(summary_embeddings @ query_embedding))

    print(f"[Retriever] Summary score: {best_summary_score:.3f} | Section score: {best_section_score:.3f}")

    # Broad if summaries win over sections by the threshold margin
    return best_summary_score > best_section_score + threshold


# ---------------------------------------------------------------------------
# 3. Query Classification — semantic match to actual sections in ChromaDB
# ---------------------------------------------------------------------------

def classify_query(query: str, collection) -> dict:
    """Semantically match the query to the sections that actually exist
    in this candidate's collection. Language agnostic — no hardcoded keywords.

    Returns a ChromaDB `where` filter dict, or {} to search everything.
    """
    results = collection.get(include=["metadatas"])
    sections = list({
        m["section"]
        for m in results["metadatas"]
        if "section" in m
    })

    if not sections:
        return {}

    query_embedding = embedder.encode([query])[0]
    section_embeddings = embedder.encode(sections)

    scores = section_embeddings @ query_embedding

    best_idx = int(np.argmax(scores))
    best_section = sections[best_idx]
    best_score = scores[best_idx]

    print(f"[Retriever] Section scores: {dict(zip(sections, scores.round(3)))}")
    print(f"[Retriever] Best section: '{best_section}' (score: {best_score:.3f})")

    if best_score > 0.3:
        return {"section": best_section}
    return {}


# ---------------------------------------------------------------------------
# 4. Fusion Retrieval — BM25 + Vector search merged with RRF
# ---------------------------------------------------------------------------

def bm25_search(query: str, chunks: list[str], top_k: int = 10) -> list[str]:
    """Keyword-based search using BM25.

    Finds chunks that contain exact or near-exact terms from the query.
    Complements vector search which finds semantic similarity.
    """
    if not chunks:
        return []

    tokenized_corpus = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = scores.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices if scores[i] > 0]


def rrf_fusion(vector_chunks: list[str], bm25_chunks: list[str], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion — merges two ranked lists into one.

    Each chunk gets a score of 1/(k + rank) from each list.
    Chunks appearing high in both lists get the highest combined score.

    k=60 is the standard value from the original RRF paper.
    """
    scores: dict[str, float] = {}

    for rank, chunk in enumerate(vector_chunks):
        scores[chunk] = scores.get(chunk, 0) + 1 / (k + rank + 1)

    for rank, chunk in enumerate(bm25_chunks):
        scores[chunk] = scores.get(chunk, 0) + 1 / (k + rank + 1)

    return sorted(scores, key=scores.get, reverse=True)


# ---------------------------------------------------------------------------
# 5. Re-ranking — score every candidate chunk with a Cross-Encoder
# ---------------------------------------------------------------------------

def rerank(query: str, chunks: list[str], top_k: int = 3) -> list[str]:
    """Re-rank `chunks` by relevance to `query` using a Cross-Encoder.

    Returns the top_k most relevant chunks in descending relevance order.
    """
    if not chunks:
        return []

    pairs = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)

    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


# ---------------------------------------------------------------------------
# 6. Main retrieve pipeline: route → expand → classify → fuse → re-rank
# ---------------------------------------------------------------------------

def retrieve(query: str, candidate_id: str, top_k: int = 3) -> list[str]:
    """Full advanced retrieval pipeline.

    Steps:
        1. Route — broad query → summary index, specific → vector index.
        2. Expand the query into multiple variations via Ollama.
        3. Classify the query → find the most relevant section semantically.
        4. For each variation, fetch chunks via vector search (filtered by section).
        5. Run BM25 keyword search on the same candidate chunks.
        6. Fuse vector and BM25 results using RRF.
        7. Re-rank fused candidates with a Cross-Encoder.
        8. Return the top_k best chunks.
    """
    collection = client.get_or_create_collection(name=candidate_id)

    # --- Step 1: Route — summary index for broad queries ---
    if is_broad_query(query, candidate_id):
        print(f"[Retriever] Broad query detected → searching summary index")
        summary_collection = client.get_or_create_collection(f"{candidate_id}_summaries")
        q_embedding = embedder.encode([query]).tolist()
        results = summary_collection.query(query_embeddings=q_embedding, n_results=top_k)
        summaries = results["documents"][0]
        print(f"[Retriever] Returning {len(summaries)} summaries")
        return summaries

    # --- Step 2: Query Expansion ---
    queries = expand_query(query)
    print(f"[Retriever] Expanded '{query}' → {queries}")

    # --- Step 3: Classify → get metadata filter ---
    metadata_filter = classify_query(query, collection)
    print(f"[Retriever] Metadata filter: {metadata_filter}")

    # --- Step 4: Vector search for every query variation ---
    fetch_per_query = 10
    vector_chunks: list[str] = []
    seen: set[str] = set()

    for q in queries:
        q_embedding = embedder.encode([q]).tolist()
        results = collection.query(
            query_embeddings=q_embedding,
            n_results=fetch_per_query,
            where=metadata_filter if metadata_filter else None,
        )
        for chunk in results["documents"][0]:
            if chunk not in seen:
                seen.add(chunk)
                vector_chunks.append(chunk)

    print(f"[Retriever] Vector search → {len(vector_chunks)} unique chunks")

    # --- Step 5: BM25 search on the same filtered chunks ---
    all_docs = collection.get(
        where=metadata_filter if metadata_filter else None,
        include=["documents"]
    )
    all_chunks_in_section = all_docs["documents"]

    bm25_chunks = bm25_search(query, all_chunks_in_section, top_k=fetch_per_query)
    print(f"[Retriever] BM25 search → {len(bm25_chunks)} chunks")

    # --- Step 6: Fuse vector + BM25 results with RRF ---
    fused = rrf_fusion(vector_chunks, bm25_chunks)
    print(f"[Retriever] RRF fusion → {len(fused)} chunks")

    # --- Step 7: Re-rank and return the best ---
    best = rerank(query, fused, top_k=top_k)
    print(f"[Retriever] Re-ranked → returning top {len(best)} chunks")
    return best