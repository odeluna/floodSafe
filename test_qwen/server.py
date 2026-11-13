import json
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ollama import Client

# ---- Config -----------------------------------------------------------
OLLAMA_HOST = "http://10.184.37.28:11434/"
MODEL = "qwen3-vl:8b"  # same model name as your test_quen.py
AREA_IDS = ["leon", "olmos", "leonv", "apache", "med", "utsa", "helotes"]
# -----------------------------------------------------------------------

client = Client(host=OLLAMA_HOST)
app = FastAPI()

# Only matters if the HTML is opened from somewhere other than this server
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def call_model(prompt, as_json=False):
    kwargs = {"format": "json"} if as_json else {}
    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    text = response["message"]["content"]
    # Some Qwen3 builds include <think>...</think>; strip it if present
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


@app.get("/")
def index():
    return FileResponse("index.html")


@app.post("/api/ask")
def ask(payload: dict):
    question = payload["question"]

    # 1) The text answer for the user
    answer = call_model(question)

    # 2) Flood risk scores (0-1) for each map area
    scores_prompt = (
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Rate flood risk from 0 (none) to 1 (worst) for each of these area ids: {AREA_IDS}. "
        'Reply with ONLY a JSON object like {"leon": 0.9, "utsa": 0.2}.'
    )
    areas = {}
    try:
        raw = call_model(scores_prompt, as_json=True)
        parsed = json.loads(re.search(r"\{.*\}", raw, re.S).group())
        areas = {
            k: max(0.0, min(1.0, float(v)))
            for k, v in parsed.items()
            if k in AREA_IDS
        }
    except Exception as e:
        print("Score parsing failed:", e)  # page falls back to sample severities

    # Shape matches your parseApiResponse: json.areas = { id: score }
    return {"answer": answer, "areas": areas}


# Serves the map image and other files in this folder. Must stay BELOW the routes above.
app.mount("/", StaticFiles(directory="."), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
