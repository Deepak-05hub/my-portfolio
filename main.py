from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import json
import os
import random
import re

app = FastAPI()

# ─────────────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────
# CONNECT GOOGLE GEMINI — with startup validation
# ─────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    print("❌ CRITICAL: GEMINI_API_KEY environment variable is NOT set!")
    print("   Run: export GEMINI_API_KEY='your_key_here'  then restart the server.")
else:
    print(f"✅ GEMINI_API_KEY loaded (starts with: {GEMINI_API_KEY[:8]}...)")

genai.configure(api_key=GEMINI_API_KEY)

# Use gemini-1.5-flash — most stable and widely available model
try:
    model = genai.GenerativeModel("gemini-1.5-flash")
    print("✅ Gemini model loaded: gemini-1.5-flash")
except Exception as e:
    print(f"❌ Failed to load Gemini model: {e}")
    model = None

# ─────────────────────────────────────────────────────
# LOAD PROFILE DATA
# ─────────────────────────────────────────────────────
profile_raw = {}
profile_data_str = "No profile data found."

try:
    with open("profile.json", "r") as f:
        profile_raw = json.load(f)
        profile_data_str = json.dumps(profile_raw)
    print("✅ profile.json loaded successfully.")
except FileNotFoundError:
    print("❌ profile.json not found! Make sure it is in the same folder as main.py.")
except Exception as e:
    print(f"❌ Error loading profile.json: {e}")


# ─────────────────────────────────────────────────────
# KEYWORD MATCHING ENGINE (Layer 1 — instant, no API)
# ─────────────────────────────────────────────────────
def find_keyword_answer(question: str):
    question_lower = question.lower().strip()
    keyword_qa = profile_raw.get("keyword_qa", {})

    for bucket_name, bucket_data in keyword_qa.items():
        keywords = bucket_data.get("keywords", [])
        answers  = bucket_data.get("answers", [])
        for keyword in keywords:
            if keyword.lower() in question_lower:
                if answers:
                    return random.choice(answers)
    return None


# ─────────────────────────────────────────────────────
# FAQ SIMILARITY ENGINE (Layer 2 — Jaccard matching)
# ─────────────────────────────────────────────────────
def find_faq_answer(question: str):
    question_lower = question.lower().strip()
    faq_list = profile_raw.get("faq", [])

    filler = {"what","is","are","does","do","can","how","tell","me","about",
              "the","a","an","your","his","deepak","i","you","please","would",
              "could","should","know","have","has","my","their","its"}

    def clean(text):
        words = re.sub(r"[^a-z0-9 ]", "", text.lower()).split()
        return set(w for w in words if w not in filler)

    q_words = clean(question_lower)
    best_match = None
    best_score = 0

    for faq in faq_list:
        faq_words = clean(faq.get("question", ""))
        if not faq_words:
            continue
        intersection = q_words & faq_words
        union = q_words | faq_words
        score = len(intersection) / len(union) if union else 0
        if score > best_score:
            best_score = score
            best_match = faq.get("answer", "")

    if best_score >= 0.35 and best_match:
        return best_match
    return None


# ─────────────────────────────────────────────────────
# CLEAN RESPONSE — strip all markdown for voice output
# ─────────────────────────────────────────────────────
def clean_response(text: str) -> str:
    text = text.replace("**", "").replace("*", "").replace("#", "")
    text = text.replace("`", "").replace("~", "")
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────
# GEMINI CALL — isolated function with full error detail
# ─────────────────────────────────────────────────────
def call_gemini(user_question: str) -> str:
    """
    Calls Gemini API. Returns clean text answer or raises exception with detail.
    """
    if model is None:
        raise RuntimeError("Gemini model was not initialized. Check your API key.")

    prompt = f"""You are the personal AI voice assistant for DEEPAK L's portfolio website.
Speak in a friendly, confident, conversational tone.
Keep every response to 2-3 sentences max — short and easy to speak out loud.
No markdown, no asterisks, no bullet points, no bold. Plain text only.
Never return an empty response.

Here is Deepak's full profile for reference:
{profile_data_str}

RULES:
- For questions about Deepak: use his profile data above as truth.
- For general tech questions (AI, Python, SQL, ML, React, Git, etc): answer clearly and simply.
- For greetings: introduce yourself as Deepak's AI assistant.
- For unrelated topics (sports, cooking, movies): say "That is outside my area, but ask me about Deepak or tech!"
- Never say "I don't know" — always give something useful.

User Question: {user_question}

Answer now in plain conversational text only. 2-3 sentences max."""

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=200,
        )
    )

    # Extract text from response — handle all possible response shapes
    text = ""

    if hasattr(response, "text") and response.text:
        text = response.text

    elif hasattr(response, "parts") and response.parts:
        text = " ".join(p.text for p in response.parts if hasattr(p, "text") and p.text)

    elif hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
            text = " ".join(
                p.text for p in candidate.content.parts
                if hasattr(p, "text") and p.text
            )

    if not text:
        # Log full response object for debugging
        print(f"⚠️ Empty Gemini response. Full object: {response}")
        raise ValueError("Gemini returned an empty response.")

    return clean_response(text)


# ─────────────────────────────────────────────────────
# MAIN /ask ENDPOINT
# Three layers: keyword → FAQ → Gemini
# ─────────────────────────────────────────────────────
@app.post("/ask")
async def ask_question(request: Request):
    try:
        body = await request.json()
        user_question = body.get("question", "").strip()

        if not user_question:
            return {"answer": "I did not catch that. Could you ask your question again?", "source": "guard"}

        print(f"\n📥 Question: {user_question}")

        # Layer 1: Keyword match
        answer = find_keyword_answer(user_question)
        if answer:
            print("✅ Source: keyword")
            return {"answer": answer, "source": "keyword"}

        # Layer 2: FAQ match
        answer = find_faq_answer(user_question)
        if answer:
            print("✅ Source: faq")
            return {"answer": answer, "source": "faq"}

        # Layer 3: Gemini AI
        print("🤖 Calling Gemini...")
        answer = call_gemini(user_question)
        print("✅ Source: gemini")
        return {"answer": answer, "source": "gemini"}

    except genai.types.BlockedPromptException as e:
        print(f"🚫 Gemini blocked the prompt: {e}")
        return {"answer": "I was not able to answer that question. Try rephrasing it!", "source": "blocked"}

    except Exception as e:
        # Print the REAL error so you can see it in your terminal
        import traceback
        print(f"\n❌ REAL ERROR in /ask:")
        print(traceback.format_exc())
        return {
            "answer": "I ran into an issue. Please check the server terminal for the exact error.",
            "source": "error",
            "debug_error": str(e)  # visible in API response for easier debugging
        }


# ─────────────────────────────────────────────────────
# UTILITY ENDPOINTS
# ─────────────────────────────────────────────────────
@app.get("/")
async def root():
    api_key_status = "SET" if GEMINI_API_KEY else "MISSING — THIS IS WHY AI IS FAILING"
    profile_status = "LOADED" if profile_raw else "MISSING"
    model_status   = "READY" if model else "FAILED TO LOAD"
    return {
        "status": "Deepak AI Backend Running",
        "version": "3.1",
        "gemini_api_key": api_key_status,
        "profile_json": profile_status,
        "gemini_model": model_status,
        "model_name": "gemini-1.5-flash",
        "tip": "If AI fails, visit /debug to see full diagnostics"
    }


@app.get("/debug")
async def debug():
    """Full diagnostic — visit this URL to find the exact problem."""
    api_key_set = bool(GEMINI_API_KEY)
    profile_loaded = bool(profile_raw)
    model_ready = model is not None

    # Try a live Gemini test call
    gemini_test = "not tested"
    gemini_error = None
    if model_ready and api_key_set:
        try:
            r = model.generate_content("Say the word OK only.")
            gemini_test = "SUCCESS — Gemini is working"
        except Exception as e:
            gemini_test = "FAILED"
            gemini_error = str(e)

    return {
        "checks": {
            "1_api_key_set":       "✅ YES" if api_key_set      else "❌ NO — set GEMINI_API_KEY env variable",
            "2_profile_loaded":    "✅ YES" if profile_loaded   else "❌ NO — profile.json missing or broken",
            "3_model_initialized": "✅ YES" if model_ready      else "❌ NO — model failed to initialize",
            "4_gemini_live_test":  gemini_test,
        },
        "gemini_error_detail": gemini_error,
        "api_key_preview":     GEMINI_API_KEY[:8] + "..." if api_key_set else "NOT SET",
        "profile_keys":        list(profile_raw.keys()) if profile_loaded else [],
        "keyword_buckets":     len(profile_raw.get("keyword_qa", {})),
        "faq_count":           len(profile_raw.get("faq", [])),
        "fix_instructions": {
            "missing_api_key": "Run: export GEMINI_API_KEY='your_key'  then restart uvicorn",
            "wrong_model":     "Model is now gemini-1.5-flash (more stable than 2.5-flash)",
            "missing_profile": "Make sure profile.json is in the same folder as main.py"
        }
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": "gemini-1.5-flash"}


@app.post("/test")
async def test(request: Request):
    """See which engine answers a question without actually calling Gemini."""
    data = await request.json()
    q = data.get("question", "").strip()
    return {
        "question": q,
        "keyword_engine": find_keyword_answer(q) or "no match",
        "faq_engine":     find_faq_answer(q)     or "no match",
        "gemini_needed":  find_keyword_answer(q) is None and find_faq_answer(q) is None
    }
