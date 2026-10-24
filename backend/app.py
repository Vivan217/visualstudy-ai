import os, json, base64
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB image limit

PROMPT = """You are a friendly teacher. Analyse this educational diagram.
Write ALL text values in {lang}. Difficulty level: {level}.
Return ONLY one JSON object, no markdown, in exactly this shape:
{{"topic": "string",
 "summary": "string",
 "components": [{{"name": "string", "explanation": "string"}}],
 "quiz": [{{"q": "string", "options": ["a","b","c","d"], "answer": 0, "why": "string"}}],
 "notes": ["string"],
 "uncertain": ["string"]}}
Rules: make exactly 5 quiz questions, "answer" is the index (0-3) of the correct option.
In "uncertain" list anything you cannot read clearly or are not sure about.
Never invent labels that are not visible in the image."""


def error_message(r):
    try:
        return r.json()["error"]["message"]
    except Exception:
        return r.text[:300]


def call_gemini(img_b64, mime, prompt):
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or "yahan" in key.lower():
        raise RuntimeError("GEMINI_API_KEY .env me set nahi hai. backend/.env me asli key daalo aur server restart karo.")
    model = os.getenv("GEMINI_MODEL", "gemma-4-26b-a4b-it").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    body = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": mime, "data": img_b64}},
        ]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }
    r = requests.post(url, headers=headers, json=body, timeout=120)
    if r.status_code == 400:
        # some models do not support JSON mode: retry without it
        body.pop("generationConfig")
        r = requests.post(url, headers=headers, json=body, timeout=120)
    if not r.ok:
        raise RuntimeError(f"Gemini API {r.status_code}: {error_message(r)}")
    data = r.json()
    if not data.get("candidates"):
        raise RuntimeError(f"Model ne jawab nahi diya: {data.get('promptFeedback', 'no candidates')}")
    parts = data["candidates"][0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text.strip():
        raise RuntimeError("Model ka jawab khali aaya, dobara try karo.")
    return text


def call_ollama(img_b64, mime, prompt):
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    body = {
        "model": os.getenv("OLLAMA_MODEL", "gemma4"),
        "stream": False,
        "format": "json",
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
    }
    try:
        r = requests.post(f"{host}/api/chat", json=body, timeout=300)
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama chal nahi raha. Ollama app kholo ya `ollama serve` chalao.")
    if not r.ok:
        raise RuntimeError(f"Ollama {r.status_code}: {error_message(r)}")
    return r.json()["message"]["content"]


def parse(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Model ne JSON nahi diya, dobara Analyse dabao.")
    return json.loads(text[start:end + 1])


def normalize(d):
    quiz = []
    for q in d.get("quiz") or []:
        opts = [str(o) for o in (q.get("options") or [])]
        if len(opts) < 2:
            continue
        try:
            ans = int(q.get("answer", 0))
        except (TypeError, ValueError):
            ans = 0
        quiz.append({
            "q": str(q.get("q", "")),
            "options": opts,
            "answer": ans if 0 <= ans < len(opts) else 0,
            "why": str(q.get("why", "")),
        })
    return {
        "topic": str(d.get("topic", "Diagram")),
        "summary": str(d.get("summary", "")),
        "components": [
            {"name": str(c.get("name", "")), "explanation": str(c.get("explanation", ""))}
            for c in (d.get("components") or []) if isinstance(c, dict)
        ],
        "quiz": quiz,
        "notes": [str(n) for n in (d.get("notes") or [])],
        "uncertain": [str(u) for u in (d.get("uncertain") or [])],
    }


@app.get("/")
def home():
    return jsonify({"app": "VisualStudy AI backend", "provider": os.getenv("PROVIDER", "gemini"), "status": "ok"})


@app.post("/api/analyze")
def analyze():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "Image nahi mili."}), 400
    lang = request.form.get("lang", "English")
    level = request.form.get("level", "Beginner")
    mime = f.mimetype or "image/jpeg"
    img_b64 = base64.b64encode(f.read()).decode()
    prompt = PROMPT.format(lang=lang, level=level)
    fn = call_ollama if os.getenv("PROVIDER", "gemini").lower() == "ollama" else call_gemini
    try:
        return jsonify(normalize(parse(fn(img_b64, mime, prompt))))
    except json.JSONDecodeError:
        return jsonify({"error": "Model ka JSON galat tha, dobara Analyse dabao."}), 502
    except requests.exceptions.Timeout:
        return jsonify({"error": "Model ne time out kiya, dobara try karo."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
