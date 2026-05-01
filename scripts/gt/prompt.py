"""
Ground Truth Prompt Generator.

Generates one or more copy-paste prompts for an annotator LLM to create
self-contained, leaf-anchored ground truth question-answer pairs for
Indonesian legal retrieval evaluation.

Anchors at the finest granularity (rincian index) so that evaluation
at coarser levels (ayat, pasal) can be derived by rolling UP to parents.

Long documents are split into multiple prompt files automatically so the
annotator always sees full node text. Leaf nodes are never truncated and
never split across prompt parts.

Per design v3 (3-type stratified): supported query types are factual,
paraphrased, multihop. Provenance dipusatkan di data/gt_provenance.json
(prompt SHA-8 per type + default models), bukan per-file sidecar.

Usage:
    python scripts/gt/prompt.py uu-13-2025
    python scripts/gt/prompt.py uu-13-2025 --questions 3
    python scripts/gt/prompt.py uu-13-2025 --type paraphrased --questions 3
    python scripts/gt/prompt.py uu-13-2025 --type multihop --questions 1
    python scripts/gt/prompt.py uu-13-2025 --stdout
    python scripts/gt/prompt.py --list
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path

# Force UTF-8 output on Windows (navigation paths contain Unicode chars like em-dash)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_INDEX = Path("data/index_rincian")
TMP_DIR = Path("tmp")
SELECTION_FILE = Path("data/gt_doc_selection.json")
PROMPTS_DIR = Path("scripts/gt/prompts")
PROVENANCE_FILE = Path("data/gt_provenance.json")
PROVENANCE_SCHEMA_VERSION = "1.0"
DEFAULT_ANNOTATOR = "claude-sonnet-4-6"
DEFAULT_JUDGE = "gpt-5"
DEFAULT_PROMPT_CHAR_BUDGET = 45000
PROMPT_BUDGET_GROWTH = 1.25

VALID_QUERY_TYPES = {"factual", "paraphrased", "multihop"}

# Preamble section names to exclude from GT (they are not substantive law text)
PREAMBLE_KEYWORDS = ["Menimbang", "Mengingat", "Menetapkan", "Pembukaan"]

# Minimum body leaf nodes required for GT generation (short docs produce unavoidable duplicates)
MIN_LEAF_FOR_GT = 5


def load_template(query_type: str) -> str:
    """Load a prompt template file from scripts/gt/prompts/<type>.txt."""
    path = PROMPTS_DIR / f"{query_type}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt template for query_type='{query_type}' not found at {path}. "
            f"Expected one of: {sorted(VALID_QUERY_TYPES)}"
        )
    return path.read_text(encoding="utf-8")


def prompt_template_version(query_type: str = "factual") -> str:
    """Return the SHA-8 hash of the current prompt template for provenance."""
    template = load_template(query_type)
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:8]


def raw_filename(doc_id: str, query_type: str) -> str:
    """Return the raw GT filename including query type tag."""
    return f"{doc_id}__{query_type}.json"


def update_provenance() -> Path:
    """Refresh data/gt_provenance.json with current prompt SHA-8s.

    Idempotent. Loads existing file or initializes from defaults, recomputes
    prompt_versions for all valid query types from current template files,
    bumps last_updated only when something changed.
    """
    PROVENANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if PROVENANCE_FILE.exists():
        with open(PROVENANCE_FILE, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "last_updated": "",
            "prompt_versions": {},
            "models": {
                "annotator": DEFAULT_ANNOTATOR,
                "judge": DEFAULT_JUDGE,
            },
            "overrides": [],
        }

    current = {qt: prompt_template_version(qt) for qt in sorted(VALID_QUERY_TYPES)}
    if data.get("prompt_versions") != current:
        data["prompt_versions"] = current
        data["last_updated"] = dt.datetime.now().isoformat(timespec="seconds")
        data.setdefault("schema_version", PROVENANCE_SCHEMA_VERSION)
        data.setdefault("models", {"annotator": DEFAULT_ANNOTATOR, "judge": DEFAULT_JUDGE})
        data.setdefault("overrides", [])
        with open(PROVENANCE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return PROVENANCE_FILE


def load_selection_for(category: str) -> set[str] | None:
    """Return the selected doc_id set for a category, or None if absent."""
    if not SELECTION_FILE.exists():
        return None
    with open(SELECTION_FILE, encoding="utf-8") as f:
        data = json.load(f)
    entry = data.get(category)
    if not entry:
        return None
    return {row["doc_id"] for row in entry.get("selected", [])}


def default_output_path(doc_id: str, query_type: str = "factual") -> Path:
    """Return the default single-prompt path under the repo-local tmp folder."""
    return TMP_DIR / f"gt_{doc_id}__{query_type}.txt"


def default_output_prefix(doc_id: str, query_type: str = "factual") -> Path:
    """Return the default multipart prefix under the repo-local tmp folder."""
    return TMP_DIR / f"gt_{doc_id}__{query_type}"


def manifest_path_from_prefix(prefix: Path) -> Path:
    """Return the manifest path for a multipart prompt run."""
    return prefix.parent / f"{prefix.name}_manifest.json"


def part_path_from_prefix(prefix: Path, part_index: int) -> Path:
    """Return the prompt file path for one part."""
    return prefix.parent / f"{prefix.name}_part{part_index:02d}.txt"


def make_output_target(doc_id: str, out_arg: str | None, multipart: bool, query_type: str = "factual") -> Path:
    """Resolve output file/prefix for single or multipart prompt generation."""
    if not out_arg:
        if multipart:
            return default_output_prefix(doc_id, query_type)
        return default_output_path(doc_id, query_type)

    out_path = Path(out_arg)
    if multipart:
        return out_path.parent / out_path.stem
    return out_path


def find_doc(doc_id: str) -> Path | None:
    """Search for doc_id across all category subfolders."""
    for path in DATA_INDEX.rglob("*.json"):
        if path.stem == doc_id and path.name != "catalog.json":
            return path
    return None


def collect_leaf_nodes(
    nodes: list[dict],
    results: list[dict] | None = None,
    parent_path: str = "",
) -> list[dict]:
    """Recursively collect all leaf nodes (nodes with text content) from tree."""
    if results is None:
        results = []
    for node in nodes:
        node_path = node.get("navigation_path", "").strip()
        if not node_path and parent_path and node.get("title"):
            node_path = f"{parent_path} > {node['title']}"
        elif not node_path:
            node_path = parent_path

        if "nodes" in node and node["nodes"]:
            collect_leaf_nodes(node["nodes"], results, parent_path=node_path)
        elif node.get("text"):
            leaf = {
                "node_id": node["node_id"],
                "title": node.get("title", ""),
                "navigation_path": node_path,
                "text": node["text"].strip(),
            }
            if node.get("penjelasan"):
                leaf["penjelasan"] = node["penjelasan"].strip()
            results.append(leaf)
    return results


def filter_preamble(leaves: list[dict]) -> list[dict]:
    """Remove preamble leaves (Menimbang, Mengingat, Menetapkan, Pembukaan)."""
    filtered = [
        leaf for leaf in leaves
        if not any(kw in leaf.get("navigation_path", "") for kw in PREAMBLE_KEYWORDS)
    ]
    return filtered if filtered else leaves


def compute_adaptive_n(leaf_count: int) -> int:
    """
    Compute the adaptive question count based on number of rincian-index leaf nodes.

    Returns 0 if leaf_count < MIN_LEAF_FOR_GT, doc too small for meaningful GT.
    Otherwise returns min(leaf_count, 5): cap at 5 keeps annotation diverse;
    n <= leaf_count guarantees no forced anchor reuse.
    """
    if leaf_count < MIN_LEAF_FOR_GT:
        return 0
    return min(leaf_count, 5)


def section_name(leaf: dict) -> str:
    """Return the top-level section for a leaf node."""
    nav = leaf.get("navigation_path", "")
    if " > " in nav:
        return nav.split(" > ")[0].strip()
    return "Lainnya"


def render_leaf_block(leaf: dict) -> str:
    """Render one full, untruncated leaf block for the prompt."""
    block = (
        f"[node_id: {leaf['node_id']}]\n"
        f"Judul: {leaf['title']}\n"
        f"Path: {leaf['navigation_path']}\n"
        f"Teks:\n{leaf['text']}"
    )
    penjelasan = leaf.get("penjelasan", "")
    if penjelasan and penjelasan.strip().lower().rstrip(".") != "cukup jelas":
        block += f"\nPenjelasan:\n{penjelasan}"
    return block


def render_grouped_blocks(leaf_nodes: list[dict]) -> str:
    """
    Render full ayat-index leaves with section headers inserted inline,
    preserving document order (depth-first traversal order from the index).

    A new "--- SECTION ---" header is emitted whenever the section name changes.
    This avoids reordering nodes (e.g. single-leaf pasals would otherwise all
    be batched into one "Lainnya" block that appears before multi-ayat pasals).
    """
    parts: list[str] = []
    current_section: str | None = None
    for leaf in leaf_nodes:
        sec = section_name(leaf)
        if sec != current_section:
            parts.append(f"\n--- {sec} ---\n")
            current_section = sec
        parts.append(render_leaf_block(leaf))
    return "\n".join(parts)


def render_part_header(part_index: int, total_parts: int, quota: int, leaf_count: int) -> str:
    """Render part metadata shown to the annotator."""
    if total_parts <= 1:
        return ""
    return (
        "=== PART INFO ===\n"
        f"Part: {part_index} of {total_parts}\n"
        f"Question quota for this part: {quota}\n"
        f"Leaf nodes in this part: {leaf_count}\n\n"
    )


def build_prompt(
    doc: dict,
    leaf_nodes: list[dict],
    n_questions: int,
    part_index: int = 1,
    total_parts: int = 1,
    query_type: str = "factual",
) -> str:
    """Build one prompt for a specific part using the per-type template."""
    template = load_template(query_type)
    fmt_kwargs = {
        "judul": doc["judul"],
        "doc_id": doc["doc_id"],
        "part_header": render_part_header(part_index, total_parts, n_questions, len(leaf_nodes)),
        "leaf_blocks_grouped": render_grouped_blocks(leaf_nodes),
        "N": n_questions,
    }
    return template.format(**fmt_kwargs)


def pack_prompt_parts(
    doc: dict,
    leaf_nodes: list[dict],
    total_questions: int,
    base_budget: int = DEFAULT_PROMPT_CHAR_BUDGET,
    query_type: str = "factual",
) -> tuple[list[list[dict]], int]:
    """Split the doc into prompt parts using whole-node packing."""
    if not leaf_nodes:
        return [[]], base_budget

    budget = base_budget
    parts: list[list[dict]] = []

    while True:
        parts = []
        current: list[dict] = []
        for leaf in leaf_nodes:
            trial = current + [leaf]
            trial_prompt = build_prompt(doc, trial, n_questions=1, query_type=query_type)
            if current and len(trial_prompt) > budget:
                parts.append(current)
                current = [leaf]
            else:
                current = trial
        if current:
            parts.append(current)

        if len(parts) <= max(total_questions, 1):
            return parts, budget
        budget = math.ceil(budget * PROMPT_BUDGET_GROWTH)


def allocate_question_quotas(parts: list[list[dict]], total_questions: int) -> list[int]:
    """Allocate total question count proportionally across prompt parts."""
    if not parts:
        return []
    if len(parts) == 1:
        return [total_questions]

    leaf_counts = [len(part) for part in parts]
    total_leaves = sum(leaf_counts)
    quotas = [1 for _ in parts]
    remaining = total_questions - len(parts)
    if remaining < 0:
        raise ValueError("Cannot allocate at least one question per part")

    if remaining == 0:
        return quotas

    raw_extra = [remaining * (count / total_leaves) for count in leaf_counts]
    base_extra = [math.floor(x) for x in raw_extra]
    quotas = [q + extra for q, extra in zip(quotas, base_extra)]
    assigned = sum(quotas)
    leftovers = total_questions - assigned

    remainders = sorted(
        ((raw_extra[i] - base_extra[i], i) for i in range(len(parts))),
        reverse=True,
    )
    for _, idx in remainders[:leftovers]:
        quotas[idx] += 1

    return quotas


def build_prompt_parts(
    doc: dict,
    n_questions: int,
    char_budget: int = DEFAULT_PROMPT_CHAR_BUDGET,
    query_type: str = "factual",
) -> tuple[list[dict], int]:
    """Build single or multipart prompt payloads for one document."""
    leaves = collect_leaf_nodes(doc["structure"])
    leaf_nodes = filter_preamble(leaves)

    parts, final_budget = pack_prompt_parts(
        doc, leaf_nodes, n_questions, base_budget=char_budget, query_type=query_type,
    )
    quotas = allocate_question_quotas(parts, n_questions)
    total_parts = len(parts)

    prompt_parts = []
    for idx, (part_leaves, quota) in enumerate(zip(parts, quotas), start=1):
        prompt_parts.append({
            "part_index": idx,
            "total_parts": total_parts,
            "question_quota": quota,
            "leaf_count": len(part_leaves),
            "node_ids": [leaf["node_id"] for leaf in part_leaves],
            "prompt": build_prompt(
                doc, part_leaves, n_questions=quota,
                part_index=idx, total_parts=total_parts,
                query_type=query_type,
            ),
        })
    return prompt_parts, final_budget


def write_single_prompt(output_path: Path, prompt: str) -> None:
    """Write one prompt file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt, encoding="utf-8")


def write_multipart_prompts(prefix: Path, doc: dict, prompt_parts: list[dict], total_questions: int, final_budget: int, category: str = "") -> tuple[list[Path], Path]:
    """Write multipart prompt files plus a manifest."""
    prefix.parent.mkdir(parents=True, exist_ok=True)

    part_paths = []
    for part in prompt_parts:
        path = part_path_from_prefix(prefix, part["part_index"])
        path.write_text(part["prompt"], encoding="utf-8")
        part_paths.append(path)

    manifest = {
        "doc_id": doc["doc_id"],
        "judul": doc["judul"],
        "total_parts": len(prompt_parts),
        "total_questions": total_questions,
        "prompt_char_budget": final_budget,
        "parts": [
            {
                "part_index": part["part_index"],
                "question_quota": part["question_quota"],
                "leaf_count": part["leaf_count"],
                "node_ids": part["node_ids"],
                "prompt_file": str(path),
                "expected_raw_part_file": f"data/ground_truth_parts/{category}/{doc['doc_id']}/part{part['part_index']:02d}.json" if category else f"data/ground_truth_parts/{doc['doc_id']}/part{part['part_index']:02d}.json",
            }
            for part, path in zip(prompt_parts, part_paths)
        ],
    }
    manifest_path = manifest_path_from_prefix(prefix)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return part_paths, manifest_path


def list_available_docs() -> None:
    """Print all available doc_ids from the index."""
    docs = []
    for path in DATA_INDEX.rglob("*.json"):
        if path.name != "catalog.json":
            docs.append((path.parent.name, path.stem))
    docs.sort()
    print(f"\nDokumen tersedia di {DATA_INDEX}:\n")
    current_cat = None
    for cat, doc_id in docs:
        if cat != current_cat:
            print(f"\n  [{cat}]")
            current_cat = cat
        print(f"    {doc_id}")
    print(f"\nTotal: {len(docs)} dokumen\n")


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description="Generate leaf-anchored GT prompt for the annotator LLM")
    ap.add_argument("doc_id", nargs="?", help="Document ID (e.g. uu-13-2025)")
    ap.add_argument(
        "--questions", "-q", type=int, default=None,
        help="Override question count (default: adaptive min(leaf_count, 5), skips docs with < 5 body leaves)",
    )
    ap.add_argument(
        "--out", "-o", type=str, default=None,
        help="Save prompt to file; for multipart output this acts as a filename prefix",
    )
    ap.add_argument(
        "--stdout", action="store_true",
        help="Print prompt to stdout (single-prompt docs only)",
    )
    ap.add_argument(
        "--list", "-l", action="store_true",
        help="List all available doc_ids",
    )
    ap.add_argument(
        "--char-budget", type=int, default=DEFAULT_PROMPT_CHAR_BUDGET,
        help=f"Approximate prompt-size budget before automatic multipart splitting (default: {DEFAULT_PROMPT_CHAR_BUDGET})",
    )
    ap.add_argument(
        "--allow-unselected", action="store_true",
        help="Allow generating a prompt for a doc that is not in data/gt_doc_selection.json",
    )
    ap.add_argument(
        "--type", "-t", type=str, default="factual",
        choices=sorted(VALID_QUERY_TYPES),
        help="Query type to generate (factual, paraphrased, multihop). Default: factual.",
    )
    args = ap.parse_args()

    if args.list or not args.doc_id:
        list_available_docs()
        if not args.doc_id:
            ap.print_help()
        return

    doc_path = find_doc(args.doc_id)
    if not doc_path:
        print(f"ERROR: doc_id '{args.doc_id}' tidak ditemukan di {DATA_INDEX}")
        print("Gunakan --list untuk melihat semua doc_id yang tersedia.")
        sys.exit(1)

    category = doc_path.parent.name
    selected = load_selection_for(category)
    if selected is not None and args.doc_id not in selected and not args.allow_unselected:
        print(f"ERROR: doc '{args.doc_id}' is NOT in {SELECTION_FILE} for category {category}")
        print("It will act as a distractor in the corpus, not a GT source.")
        print("If this is intentional, pass --allow-unselected.")
        sys.exit(1)

    with open(doc_path, encoding="utf-8") as f:
        doc = json.load(f)

    query_type = args.type

    leaves = collect_leaf_nodes(doc["structure"])
    leaf_nodes = filter_preamble(leaves)
    adaptive_n = compute_adaptive_n(len(leaf_nodes))
    n_used = args.questions if args.questions is not None else adaptive_n

    if n_used == 0:
        print(f"\nDokumen    : {doc['judul'][:80]}")
        print(f"doc_id     : {doc['doc_id']}")
        print(f"Leaf nodes : {len(leaf_nodes)} body leaf nodes (setelah filter preamble)")
        if len(leaf_nodes) == 0:
            print(f"\n[SKIP] Tidak ada body leaf nodes, kemungkinan hanya preamble.")
        else:
            print(f"\n[SKIP] Hanya {len(leaf_nodes)} leaf nodes (< {MIN_LEAF_FOR_GT}); "
                  f"dokumen terlalu kecil untuk dijadikan GT.")
        sys.exit(0)

    prompt_parts, final_budget = build_prompt_parts(
        doc, n_questions=n_used, char_budget=args.char_budget, query_type=query_type,
    )
    multipart = len(prompt_parts) > 1

    override = f" override" if args.questions is not None else ""
    mode = "multipart" if multipart else "single"
    print(f"\n{doc['doc_id']} type={query_type}, "
          f"{len(leaf_nodes)} leaves, N={n_used}{override} (adaptive {adaptive_n}), "
          f"{mode}, budget={final_budget}")

    if multipart and args.stdout:
        print("\nERROR: Prompt multipart terlalu besar untuk --stdout.")
        print("Jalankan tanpa --stdout agar file part otomatis disimpan ke tmp/.")
        sys.exit(1)

    provenance_path = update_provenance()

    if not multipart:
        output_path = make_output_target(doc["doc_id"], args.out, multipart=False, query_type=query_type)
        raw_target_name = raw_filename(args.doc_id, query_type)
        raw_dir = Path("data/ground_truth_raw") / category
        raw_dir.mkdir(parents=True, exist_ok=True)
        placeholder = raw_dir / raw_target_name

        if args.stdout:
            print("=" * 70)
            print("COPY PROMPT DI BAWAH INI KE ANNOTATOR LLM (Claude Sonnet 4.6, cross-family dari Gemini retrieval):")
            print("=" * 70)
            print(prompt_parts[0]["prompt"])
            print("=" * 70)
            print(f"\nRaw target -> {placeholder}")
            print("\nNext.")
            print(f"  1. Paste output JSON to {placeholder}")
            print(f"  2. python scripts/gt/build_validate.py --doc-id {args.doc_id} --type {query_type}")
            print(f"  3. Paste tmp/validate_{args.doc_id}__{query_type}.txt to Judge LLM (GPT-5), paste full response over {placeholder}")
            print(f"  4. python scripts/gt/apply_validation.py --doc-id {args.doc_id} --type {query_type}")
            print(f"  5. python scripts/gt/log_review.py {args.doc_id} --type {query_type}")
            return

        write_single_prompt(output_path, prompt_parts[0]["prompt"])
        if not placeholder.exists():
            placeholder.write_text("[]", encoding="utf-8")

        print("\nCreated.")
        print(f"  Annotator prompt -> {output_path}")
        print(f"  Raw placeholder  -> {placeholder}")
        print(f"  Provenance       -> {provenance_path} (prompt SHA-8 refreshed)")
        print("\nNext.")
        print(f"  1. Paste {output_path} to Annotator LLM, paste JSON output to {placeholder}")
        print(f"  2. python scripts/gt/build_validate.py --doc-id {args.doc_id} --type {query_type}")
        print(f"  3. Paste tmp/validate_{args.doc_id}__{query_type}.txt to Judge LLM, paste full response over {placeholder}")
        print(f"  4. python scripts/gt/apply_validation.py --doc-id {args.doc_id} --type {query_type}")
        print(f"  5. python scripts/gt/log_review.py {args.doc_id} --type {query_type}")
        return

    prefix = make_output_target(doc["doc_id"], args.out, multipart=True, query_type=query_type)
    part_paths, manifest_path = write_multipart_prompts(prefix, doc, prompt_parts, total_questions=n_used, final_budget=final_budget, category=category)

    parts_basename = f"{doc['doc_id']}__{query_type}"
    parts_dir = Path("data/ground_truth_parts") / category / parts_basename
    parts_dir.mkdir(parents=True, exist_ok=True)
    for part in prompt_parts:
        ph = parts_dir / f"part{part['part_index']:02d}.json"
        if not ph.exists():
            ph.write_text("[]", encoding="utf-8")
    raw_placeholder = Path("data/ground_truth_raw") / category / raw_filename(doc["doc_id"], query_type)
    (Path("data/ground_truth_raw") / category).mkdir(parents=True, exist_ok=True)
    print("\nCreated.")
    print(f"  Multipart prompts -> {len(part_paths)} files in {prefix.parent}")
    for part, path in zip(prompt_parts, part_paths):
        print(f"    - part {part['part_index']:02d}: {path}  ({part['question_quota']} pertanyaan, {part['leaf_count']} leaf)")
    print(f"  Manifest          -> {manifest_path}")
    print(f"  Part placeholders -> {parts_dir}/part01.json ..")
    print(f"  Raw target        -> {raw_placeholder}")
    print(f"  Provenance        -> {provenance_path} (prompt SHA-8 refreshed)")
    print("\nNext.")
    print(f"  1. Paste each part to Annotator LLM, save outputs to {parts_dir}/part01.json, part02.json, ...")
    print(f"  2. python scripts/gt/merge_parts.py {doc['doc_id']} --type {query_type}")
    print(f"  3. python scripts/gt/build_validate.py --doc-id {doc['doc_id']} --type {query_type}")
    print(f"  4. Paste tmp/validate_{doc['doc_id']}__{query_type}.txt to Judge LLM, paste full response over {raw_placeholder}")
    print(f"  5. python scripts/gt/apply_validation.py --doc-id {doc['doc_id']} --type {query_type}")
    print(f"  6. python scripts/gt/log_review.py {doc['doc_id']} --type {query_type}")


if __name__ == "__main__":
    main()
