import json
import re
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from ollama import Client

# ---- Config -----------------------------------------------------------
OLLAMA_HOST = "http://10.184.37.28:11434/"
MODEL = "qwen3-vl:8b"  # same model name as your test_quen.py
AREA_IDS = ["leon", "olmos", "leonv", "apache", "med", "utsa", "helotes"]
MAX_TOKENS = 500       # lower = faster replies
KEEP_ALIVE = "30m"     # keep the model loaded between questions
# -----------------------------------------------------------------------

client = Client(host=OLLAMA_HOST)
app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def call_model(prompt):
    """One model call. Returns the reply text."""
    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        think=False,                      # skip long reasoning; if this errors, run: pip install -U ollama
        keep_alive=KEEP_ALIVE,
        options={"num_predict": MAX_TOKENS},
    )
    text = response["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


@app.get("/")
def index():
    return FileResponse("index.html")


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.post("/api/ask")
def ask(payload: dict):
    question = payload["question"]
    start = time.time()

    prompt = (
        "You are a flood-risk assistant for northwest San Antonio, Texas.\n"
        f"User question: {question}\n\n"
        "Reply with ONLY a JSON object with two keys:\n"
        '  "answer": a helpful reply in under 120 words (plain text),\n'
        f'  "areas": flood risk from 0 (none) to 1 (worst) for each of these ids: {AREA_IDS}.\n'
        'If the question is not about rain or flooding, set "areas" to {}.\n'
        'Example: {"answer": "...", "areas": {"leon": 0.9, "utsa": 0.2}}'
    )

    answer, areas = "", {}
    try:
        raw = call_model(prompt)
        parsed = json.loads(re.search(r"\{.*\}", raw, re.S).group())
        answer = str(parsed.get("answer", "")).strip()
        areas = {
            k: max(0.0, min(1.0, float(v)))
            for k, v in (parsed.get("areas") or {}).items()
            if k in AREA_IDS
        }
    except Exception as e:
        print("Parsing failed:", e)
        if not answer:
            answer = raw if "raw" in locals() else f"Error: {e}"

    print(f"/api/ask took {time.time() - start:.1f}s")
    return {"answer": answer, "areas": areas}


# Serves the map image and other files in this folder. Must stay BELOW the routes above.
app.mount("/", StaticFiles(directory="."), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)