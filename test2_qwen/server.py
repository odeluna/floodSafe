import re
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from ollama import Client

# ---- Config -----------------------------------------------------------
OLLAMA_HOST = "http://10.184.37.28:11434/"
MODEL = "qwen3-vl:8b"
MAX_TOKENS = 400       # lower = faster replies
KEEP_ALIVE = "30m"     # keep the model loaded between questions
NUM_CTX = 8192         # room for the data summary + question

BASE = Path(__file__).parent
DATASETS = {           # label shown to the model -> results file
    "Hurricane Harvey (2017)": BASE / "harvey.jsonl",
    "Hill Country floods (Jul 2025)": BASE / "hc.jsonl",
}
MAX_LISTED_RECORDS = 120   # list every image only for small datasets
# -----------------------------------------------------------------------

client = Client(host=OLLAMA_HOST)
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ======================= DATA LOADING ===================================
_cache = {"sig": None, "datasets": {}}


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # half-written last line while the GB10 is still appending
    return rows


def get_datasets():
    """Re-reads the files whenever they change, so new results appear automatically."""
    sig = tuple(
        (p.stat().st_mtime, p.stat().st_size) if p.exists() else None
        for p in DATASETS.values()
    )
    if sig != _cache["sig"]:
        _cache["datasets"] = {name: read_jsonl(p) for name, p in DATASETS.items()}
        _cache["sig"] = sig
    return _cache["datasets"]


# ======================= SUMMARY FOR THE CHATBOT ========================
def pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "n/a"


def is_ok(r):
    return r.get("mode") != "error" and r.get("passable") is not None


def summarize(name, rows):
    ok = [r for r in rows if is_ok(r)]
    lines = [
        f"## {name}",
        f"Images processed so far: {len(rows)} "
        f"({len(ok)} analyzed, {len(rows) - len(ok)} failed or timed out, not analyzed yet).",
    ]
    if not ok:
        return "\n".join(lines)

    flood = Counter(r.get("roads_flooded") for r in ok)
    not_pass = sum(1 for r in ok if r["passable"] is False)
    avg_conf = sum(r.get("confidence") or 0 for r in ok) / len(ok)
    lines += [
        f"Roads flooded per image: most={flood['most']}, some={flood['some']}, none={flood['none']}.",
        f"Judged NOT passable: {not_pass} of {len(ok)} ({pct(not_pass, len(ok))}); "
        f"passable: {len(ok) - not_pass}.",
        f"Average model confidence: {avg_conf:.2f}.",
    ]

    by_date = defaultdict(list)
    for r in ok:
        m = re.search(r"after_(\d{8})", r.get("image_id", ""))
        if m:
            by_date[m.group(1)].append(r)
    if by_date:
        lines.append("By image date (YYYYMMDD):")
        for d in sorted(by_date):
            rs = by_date[d]
            n = sum(1 for r in rs if r["passable"] is False)
            lines.append(f"  {d}: {len(rs)} images, {n} not passable ({pct(n, len(rs))})")

    if len(rows) <= MAX_LISTED_RECORDS:
        lines.append("Every analyzed image:")
        for r in ok:
            lines.append(
                f"  {r['image_id']}: roads flooded={r.get('roads_flooded')}, "
                f"passable={'yes' if r['passable'] else 'no'}, confidence={r.get('confidence')}"
            )
    return "\n".join(lines)


def records_for_question(question, datasets):
    """If the question names an image id or tile coordinates, include those records."""
    q = question.lower()
    coords = re.findall(r"\d{5}_\d{5}", q)
    hits = []
    for name, rows in datasets.items():
        if len(rows) <= MAX_LISTED_RECORDS:
            continue  # already listed in full
        for r in rows:
            iid = r.get("image_id", "")
            if iid.lower() in q or any(c in iid for c in coords):
                hits.append((name, r))
    lines = []
    for name, r in hits[:20]:
        if is_ok(r):
            lines.append(
                f"  [{name}] {r['image_id']}: roads flooded={r.get('roads_flooded')}, "
                f"passable={'yes' if r['passable'] else 'no'}, confidence={r.get('confidence')}"
            )
        else:
            lines.append(f"  [{name}] {r['image_id']}: not analyzed yet ({r.get('error')})")
    return ("Records matching the question:\n" + "\n".join(lines)) if lines else ""


def build_context(question):
    datasets = get_datasets()
    parts = [summarize(name, rows) for name, rows in datasets.items()]
    extra = records_for_question(question, datasets)
    if extra:
        parts.append(extra)
    return "\n\n".join(parts)


SYSTEM_PROMPT = (
    "You are Rain AI, a flood-data assistant. Answer ONLY from the measured data below, "
    "which comes from aerial imagery analyzed by a vision model. "
    "The analysis is still running, so the numbers are partial: say so when you give totals, "
    "percentages or rankings. Do not use outside knowledge about these events and do not invent "
    "numbers. If the data does not contain the answer, say what is missing. "
    "'Not passable' means the model judged the roads impassable. "
    "Reply in plain text, under 120 words.\n\n"
)


# ======================= ROUTES =========================================
@app.get("/")
def index():
    return FileResponse(BASE / "index.html")


@app.get("/measured")
def measured():
    return FileResponse(BASE / "map.html")


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/api/summary", response_class=PlainTextResponse)
def summary():
    """Open this in a browser to see exactly what the chatbot is given."""
    return build_context("")


@app.post("/api/ask")
def ask(payload: dict):
    question = (payload.get("question") or "").strip()
    if not question:
        return {"answer": "Please enter a question.", "response": "Please enter a question."}
    start = time.time()

    resp = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + build_context(question)},
            {"role": "user", "content": question},
        ],
        think=False,  # if this errors, run: pip install -U ollama
        keep_alive=KEEP_ALIVE,
        options={"num_predict": MAX_TOKENS, "num_ctx": NUM_CTX},
    )
    answer = re.sub(r"<think>.*?</think>", "", resp["message"]["content"], flags=re.S).strip()

    print(f"/api/ask took {time.time() - start:.1f}s")
    return {"answer": answer, "response": answer}


# Serves other files in this folder (images, etc). Must stay BELOW the routes above.
app.mount("/", StaticFiles(directory=BASE), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)