import chromadb
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ROUTER_LLM = "qwen3"

embedder = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder(RERANK_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PATH)


# ---------------------------------------------------------------------------
# 1. Query Expansion
# ---------------------------------------------------------------------------

def expand_query(original_query: str, n_variations: int = 3) -> list[str]:
    prompt = (
        f"You are a query-understanding module inside a retrieval pipeline.\n"
        f"Your generated queries will be used to search a document store using "
        f"both keyword matching (BM25) and semantic similarity (vector search).\n\n"
        f"Given the user's question, produce exactly {n_variations} search queries "
        f"that maximize the chance of retrieving the right chunks.\n\n"
        f"Think about:\n"
        f"- What words or phrases likely appear in the stored documents\n"
        f"- Include at least one short keyword-style query (for BM25)\n"
        f"- Include at least one natural-language query (for vector search)\n"
        f"- Cover different angles the answer might be described under\n\n"
        f"User question: \"{original_query}\"\n\n"
        f"Return ONLY the queries, one per line, no numbering, no explanation."
    )

    response = ollama.chat(
        model=ROUTER_LLM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response["message"]["content"].strip()
    variations = [line.strip() for line in raw.splitlines() if line.strip()]
    return [original_query] + variations[:n_variations]


# ---------------------------------------------------------------------------
# 2. Query Routing
# ---------------------------------------------------------------------------

def is_broad_query_llm(query: str) -> bool:
    prompt = (
        f"You are a query router for a recruitment search engine.\n"
        f"Classify the query as BROAD or SPECIFIC.\n\n"
        f"BROAD = questions requiring a holistic view of the candidate:\n"
        f"  - General overviews or summaries of the candidate\n"
        f"  - Evaluative judgments (overqualified? good fit? should I hire?)\n"
        f"  - Synthesis across multiple areas (unique skills, strengths)\n"
        f"  - Comparisons across domains (academic vs professional)\n\n"
        f"SPECIFIC = questions targeting a particular fact, skill, or topic:\n"
        f"  - Any named technology, tool, framework, or language\n"
        f"  - A specific project, company, role, or certification\n"
        f"  - A particular skill area or achievement\n"
        f"  - Even if phrased broadly, if it's about one topic it's SPECIFIC\n\n"
        f"Examples:\n"
        f'  "Tell me about the candidate" → BROAD\n'
        f'  "Summarize the candidate\'s profile" → BROAD\n'
        f'  "Is this candidate overqualified for a junior role?" → BROAD\n'
        f'  "Would this candidate be a good fit for a remote role in Europe?" → BROAD\n'
        f'  "What unique combination of skills does this candidate offer?" → BROAD\n'
        f'  "Summarize why I should consider this candidate" → BROAD\n'
        f'  "Compare the candidate\'s academic and professional experience" → BROAD\n'
        f'  "What projects has the candidate worked on?" → BROAD\n'
        f'  "Does he know Python?" → SPECIFIC\n'
        f'  "What cloud platforms does the candidate have experience with?" → SPECIFIC\n'
        f'  "Tell me about the fraud detection project" → SPECIFIC\n'
        f'  "What certifications do they have?" → SPECIFIC\n'
        f'  "Has the candidate worked on any NLP projects?" → SPECIFIC\n'
        f'  "What was the capstone project about?" → SPECIFIC\n'
        f'  "Does the candidate have Kubernetes experience?" → SPECIFIC\n\n'
        f'User query: "{query}"\n\n'
        f"Return ONLY the word BROAD or SPECIFIC. No other text."
    )

    response = ollama.chat(
        model=ROUTER_LLM,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response["message"]["content"].strip()
    # Handle Qwen3 thinking mode — take last non-empty line
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    classification = lines[-1].upper() if lines else "SPECIFIC"
    print(f"[Router] LLM classified query '{query}' as: {classification}")
    return "BROAD" in classification


# ---------------------------------------------------------------------------
# 3. Fusion Retrieval — BM25 + Vector search merged with RRF
# ---------------------------------------------------------------------------

def bm25_search(query: str, chunks: list[str], top_k: int = 10) -> list[str]:
    if not chunks:
        return []
    tokenized_corpus = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = scores.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices if scores[i] > 0]


def rrf_fusion(vector_chunks: list[str], bm25_chunks: list[str], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for rank, chunk in enumerate(vector_chunks):
        scores[chunk] = scores.get(chunk, 0) + 1 / (k + rank + 1)
    for rank, chunk in enumerate(bm25_chunks):
        scores[chunk] = scores.get(chunk, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


# ---------------------------------------------------------------------------
# 4. Re-ranking
# ---------------------------------------------------------------------------

def rerank(query: str, chunks: list[str], top_k: int = 3) -> list[str]:
    if not chunks:
        return []
    pairs = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


# ---------------------------------------------------------------------------
# 5. Main retrieve pipeline
# ---------------------------------------------------------------------------

def retrieve(query: str, candidate_id: str, top_k: int = 3) -> dict:
    """
    Run the full retrieval pipeline for a query.

    Returns a dict with:
        - chunks: list[str] — the retrieved text chunks
        - route: "broad" | "specific" — how the query was classified
        - expanded_queries: list[str] | None — query variations (specific only)
    """
    collection = client.get_or_create_collection(name=candidate_id)

    # --- Step 1: Route via LLM ---
    is_broad = is_broad_query_llm(query)

    if is_broad:
        print(f"[Retriever] Broad query detected → searching summary index")
        summary_collection = client.get_or_create_collection(f"{candidate_id}_summaries")
        q_embedding = embedder.encode([query]).tolist()
        results = summary_collection.query(query_embeddings=q_embedding, n_results=top_k)
        chunks = results["documents"][0] if results["documents"] and results["documents"][0] else []
        return {
            "chunks": chunks,
            "route": "broad",
            "expanded_queries": None,
        }

    # --- Step 2: Query Expansion ---
    queries = expand_query(query)

    # --- Step 3: Vector search (batched) ---
    fetch_per_query = 10
    vector_chunks: list[str] = []
    seen: set[str] = set()

    q_embeddings = embedder.encode(queries).tolist()
    results = collection.query(query_embeddings=q_embeddings, n_results=fetch_per_query)

    for chunk_list in results["documents"]:
        for chunk in chunk_list:
            if chunk not in seen:
                seen.add(chunk)
                vector_chunks.append(chunk)

    # --- Step 4: BM25 search (full collection) ---
    all_chunks = collection.get(include=["documents"])["documents"]
    bm25_chunks = bm25_search(query, all_chunks, top_k=fetch_per_query)

    # --- Step 5: Fuse vector + BM25 results with RRF ---
    fused = rrf_fusion(vector_chunks, bm25_chunks)

    # --- Step 6: Re-rank and return the best ---
    top_chunks = rerank(query, fused, top_k=top_k)

    return {
        "chunks": top_chunks,
        "route": "specific",
        "expanded_queries": queries,
    }