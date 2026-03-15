"""
Shared utilities for vector RAG retrieval pipelines.

Contains: LLM client, embedding, token tracking, answer generation, logging.
Used by retrieve_vector.py and retrieve_vector_hybrid.py.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"
COLLECTION_NAME = "law-pasal"
QDRANT_URL = "http://localhost:6333"
LOG_DIR = Path("data/retrieval_logs")


# ============================================================
# CLIENTS
# ============================================================

_client = None
_total_input_tokens = 0
_total_output_tokens = 0
_total_calls = 0


def _get_client():
    """Lazy-init Google GenAI client."""
    global _client
    if _client is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: GEMINI_API_KEY not set.")
            sys.exit(1)
        _client = genai.Client(api_key=api_key)
    return _client


def reset_token_counters():
    """Reset per-query token counters."""
    global _total_input_tokens, _total_output_tokens, _total_calls
    _total_input_tokens = 0
    _total_output_tokens = 0
    _total_calls = 0


def get_token_stats() -> dict:
    """Return current token usage stats."""
    return {
        "llm_calls": _total_calls,
        "input_tokens": _total_input_tokens,
        "output_tokens": _total_output_tokens,
        "total_tokens": _total_input_tokens + _total_output_tokens,
    }


# ============================================================
# EMBEDDING
# ============================================================

def embed_query(query: str) -> list[float]:
    """Embed a single query using gemini-embedding-001."""
    client = _get_client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[query],
    )
    return [float(x) for x in result.embeddings[0].values]


# ============================================================
# LLM
# ============================================================

def llm_call(prompt: str, max_retries: int = 3) -> dict:
    """Send prompt to Gemini 2.5 Flash, return parsed JSON."""
    global _total_input_tokens, _total_output_tokens, _total_calls
    client = _get_client()

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            break
        except Exception as e:
            err = str(e).lower()
            if ("rate" in err or "429" in err) and attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                print(f"  Rate limited, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    usage = response.usage_metadata
    _total_input_tokens += usage.prompt_token_count or 0
    _total_output_tokens += usage.candidates_token_count or 0
    _total_calls += 1

    text = response.text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    return json.loads(text)


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(query: str, results: list[dict],
                    verbose: bool = True) -> dict:
    """Generate a grounded answer from retrieved chunks.

    Prompt aligned with vectorless-rag's generate_answer for fair comparison.
    Difference: vector RAG retrieves across multiple docs, so each chunk
    includes its own doc source instead of a single doc_meta header.
    """
    if not results:
        return {"answer": "No answer generated", "cited_pasals": []}

    context_parts = []
    for r in results:
        part = f"### {r['title']}\n"
        part += f"Lokasi: {r['navigation_path']}\n"
        part += f"Sumber: {r['doc_title']}\n\n"
        part += f"Isi:\n{r['text']}\n"
        context_parts.append(part)

    context = "\n---\n".join(context_parts)

    prompt = f"""\
Kamu adalah asisten hukum Indonesia. Jawab pertanyaan berdasarkan HANYA teks Pasal yang diberikan.

Pertanyaan: {query}

Sumber hukum:
{context}

Balas dalam format JSON:
{{
  "answer": "<jawaban yang jelas dan lengkap berdasarkan teks Pasal>",
  "cited_pasals": ["Pasal X", "Pasal Y"]
}}

Aturan:
- Jawab berdasarkan teks Pasal saja, JANGAN mengarang
- Jika ada Penjelasan Resmi, gunakan untuk menginterpretasi
- Sebutkan nomor Pasal yang menjadi dasar jawaban
- Jawab dalam Bahasa Indonesia
- Kembalikan HANYA JSON
"""

    result = llm_call(prompt)

    if verbose:
        print(f"\n[Answer] {result.get('answer', '')[:300]}")
        print(f"  Cited: {result.get('cited_pasals', [])}")

    return result


# ============================================================
# LOGGING
# ============================================================

def save_log(result: dict):
    """Save retrieval result to data/retrieval_logs/."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy = result.get("strategy", "unknown").replace(" ", "_")
    log_path = LOG_DIR / f"{timestamp}_{strategy}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Log saved: {log_path.name}")
