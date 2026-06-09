from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import json
import os
import random
import re

app = FastAPI()

# ─────────────────────────────────────────────────────
# CORS — allow frontend to talk to backend freely
# ─────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────
# CONNECT GOOGLE GEMINI
# ─────────────────────────────────────────────────────
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# ─────────────────────────────────────────────────────
# LOAD PROFILE DATA (resume + FAQ + keyword QA)
# ─────────────────────────────────────────────────────
profile_raw = {}
profile_data_str = "No profile data found."

try:
    with open("profile.json", "r") as f:
        profile_raw = json.load(f)
        profile_data_str = json.dumps(profile_raw)
    print("✅ profile.json loaded successfully.")
except Exception as e:
    print(f"❌ Error loading profile.json: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD MATCHING ENGINE
# Scans user question for keywords and returns a pre-written answer instantly
# without calling Gemini API — fast, accurate, zero hallucination
# ─────────────────────────────────────────────────────────────────────────────

def find_keyword_answer(question: str) -> str | None:
    """
    Checks the user's question against all keyword buckets in profile.json.
    Returns a randomly selected pre-written answer if a keyword matches.
    Returns None if no match found — falls back to Gemini.
    """
    question_lower = question.lower().strip()
    keyword_qa = profile_raw.get("keyword_qa", {})

    for bucket_name, bucket_data in keyword_qa.items():
        keywords = bucket_data.get("keywords", [])
        answers  = bucket_data.get("answers", [])

        for keyword in keywords:
            # Match keyword as substring in the question
            if keyword.lower() in question_lower:
                if answers:
                    return random.choice(answers)

    return None  # No keyword matched — let Gemini handle it


# ─────────────────────────────────────────────────────────────────────────────
# FAQ EXACT / PARTIAL MATCH ENGINE
# Checks Deepak's static FAQ list for a close question match
# ─────────────────────────────────────────────────────────────────────────────

def find_faq_answer(question: str) -> str | None:
    """
    Checks user question against the FAQ section in profile.json.
    Returns the FAQ answer if a high-similarity match is found.
    """
    question_lower = question.lower().strip()
    faq_list = profile_raw.get("faq", [])

    # Remove common filler words for better matching
    filler = ["what", "is", "are", "does", "do", "can", "how", "tell", "me",
              "about", "the", "a", "an", "your", "his", "deepak", "i", "you",
              "please", "would", "could", "should", "know", "have", "has"]

    def clean(text):
        words = re.sub(r"[^a-z0-9 ]", "", text.lower()).split()
        return set(w for w in words if w not in filler)

    q_words = clean(question_lower)

    best_match = None
    best_score = 0

    for faq in faq_list:
        faq_q = faq.get("question", "")
        faq_a = faq.get("answer", "")
        faq_words = clean(faq_q)

        if not faq_words:
            continue

        # Jaccard similarity between question word sets
        intersection = q_words & faq_words
        union = q_words | faq_words
        score = len(intersection) / len(union) if union else 0

        if score > best_score:
            best_score = score
            best_match = faq_a

    # Return FAQ answer only if similarity is above threshold
    if best_score >= 0.35 and best_match:
        return best_match

    return None


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — The full brain for Gemini fallback
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""
You are the personal AI voice assistant embedded in DEEPAK L's portfolio website.
Your name is "Deepak's AI" and you speak in a friendly, confident, and conversational tone.
You represent Deepak L professionally to every visitor — recruiters, collaborators, or curious people.

══════════════════════════════════════════════════════
DEEPAK'S COMPLETE PROFILE DATA (your primary knowledge source):
══════════════════════════════════════════════════════
{profile_data_str}

══════════════════════════════════════════════════════
YOUR BEHAVIOR RULES:
══════════════════════════════════════════════════════

1. ABOUT DEEPAK — Resume, Background, Career Questions
   - Use his profile data above as the single source of truth.
   - Speak as his representative: "Deepak is...", "He has...", "He built..."
   - Never make up details not in the profile.
   - Cover: skills, projects, education, certifications, internship, hackathon, career goals, contact.

2. GENERAL TECH & AI QUESTIONS
   - Answer clearly in 2-3 simple, voice-friendly sentences.
   - Topics you MUST handle confidently:
     AI, Machine Learning, Deep Learning, Python, SQL, HTML/CSS, JavaScript, React, Java,
     REST APIs, FastAPI, JSON, Git, GitHub, Cloud Computing, GCP, AWS, Azure, Docker,
     IoT, Embedded Systems, Arduino, Power BI, Data Science, NLP, Neural Networks,
     Data Structures, Algorithms, OOP, Agile, Scrum, DevOps, CI/CD, CORS, Middleware,
     Web Speech API, Generative AI, LLM, UI/UX, Excel, MySQL, NoSQL, MongoDB,
     Responsive Design, Version Control, Jupyter Notebook, Google Colab,
     Scikit-learn, pandas, numpy, matplotlib, seaborn, tensorflow, pytorch,
     Cybersecurity basics, API security, JSON Web Tokens, OAuth,
     Linux basics, Terminal, Bash scripting, VS Code, IntelliJ,
     System Design basics, Microservices, Serverless, Edge Computing,
     Blockchain basics, Web3 basics, AR/VR basics.

3. GREETINGS & SMALL TALK
   - Respond warmly to Hi, Hello, Hey, What's up, Good morning, Namaste.
   - Give a brief friendly intro: who you are and what you can help with.

4. CAREER & HIRING QUESTIONS
   - Answer confidently: Deepak is available, open to relocation, and actively job-seeking.
   - Target roles: Software Engineer, Python Developer, Data Analyst, AI/ML Engineer, Frontend Dev.
   - Location: Open to Bangalore or remote. Immediate joiner.

5. PROJECT QUESTIONS
   - Explain the fire-fighting robot in detail when asked.
   - Talk about the AI voice portfolio site at deepak-l.me.
   - Mention the Power BI and MySQL data analytics projects.

6. RECRUITER QUESTIONS
   - Answer questions about hiring, salary (flexible), notice period (immediate), background checks positively.
   - Always end with Deepak's email or LinkedIn for follow-up.

7. HOW THIS AI WORKS
   - If asked how this chatbot or AI works, explain: FastAPI backend + Gemini AI + profile.json + keyword matching.
   - Mention the Web Speech API for voice.

8. OUT-OF-SCOPE QUESTIONS
   - If totally unrelated (cricket scores, cooking, movies, weather), say:
     "That is a bit outside my expertise, but I am great at answering anything about Deepak or tech topics!"

══════════════════════════════════════════════════════
STRICT RESPONSE FORMAT RULES:
══════════════════════════════════════════════════════
- Maximum 2-3 sentences per response. Short and voice-friendly.
- ZERO markdown. No asterisks, hashes, bullet points, or bold text.
- Never return empty. Always give something useful.
- Sound human, warm, and smart — never robotic.
- No corporate buzzwords like "synergy", "leverage", "utilize".
- Speak naturally, like a confident friend explaining things.
"""


# ─────────────────────────────────────────────────────────────────────────────
# CLEAN RESPONSE UTILITY
# Removes all markdown artifacts from AI responses for clean voice output
# ─────────────────────────────────────────────────────────────────────────────

def clean_response(text: str) -> str:
    """Remove all markdown formatting for clean plain-text voice output."""
    text = text.replace("**", "").replace("*", "").replace("#", "")
    text = text.replace("`", "").replace("_", " ").replace("~", "")
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # Remove markdown links
    text = re.sub(r"\n{2,}", " ", text)                     # Collapse newlines
    text = re.sub(r"\s{2,}", " ", text)                     # Collapse spaces
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN /ask ENDPOINT
# Priority: keyword match → FAQ match → Gemini AI
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
        user_question = data.get("question", "").strip()

        # Guard: empty question
        if not user_question:
            return {"answer": "I did not catch that. Could you ask your question again?",
                    "source": "guard"}

        print(f"\n📥 Question received: {user_question}")

        # ── STEP 1: Keyword Engine (instant, no API call) ──────────────────
        keyword_answer = find_keyword_answer(user_question)
        if keyword_answer:
            print(f"✅ Answered via keyword match")
            return {
                "answer": keyword_answer,
                "source": "keyword"
            }

        # ── STEP 2: FAQ Engine (partial match scoring) ─────────────────────
        faq_answer = find_faq_answer(user_question)
        if faq_answer:
            print(f"✅ Answered via FAQ match")
            return {
                "answer": faq_answer,
                "source": "faq"
            }

        # ── STEP 3: Gemini AI Fallback (for anything not pre-answered) ──────
        print(f"🤖 Calling Gemini AI...")

        full_prompt = f"""
{SYSTEM_PROMPT}

══════════════════════════════════════════════════════
USER'S QUESTION:
══════════════════════════════════════════════════════
{user_question}

Respond now. Follow all rules. Plain text only. 2-3 sentences max.
"""

        response = model.generate_content(full_prompt)

        # Extract text safely from Gemini response
        answer_text = ""

        if response and hasattr(response, "text") and response.text:
            answer_text = response.text

        elif response and hasattr(response, "parts") and response.parts:
            answer_text = " ".join(
                part.text for part in response.parts
                if hasattr(part, "text") and part.text
            )

        elif response and hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                answer_text = " ".join(
                    part.text for part in candidate.content.parts
                    if hasattr(part, "text") and part.text
                )

        if answer_text:
            clean_answer = clean_response(answer_text)
            print(f"✅ Gemini responded successfully")
            return {
                "answer": clean_answer,
                "source": "gemini"
            }
        else:
            print("⚠️ Gemini returned empty response")
            return {
                "answer": "I could not process that one. Try asking something about Deepak's skills, projects, or a tech concept!",
                "source": "fallback"
            }

    except json.JSONDecodeError:
        print("❌ Invalid JSON in request body")
        return {"answer": "That request did not look right. Please send a valid question.", "source": "error"}

    except Exception as e:
        print(f"❌ Backend Error: {type(e).__name__}: {e}")
        return {
            "answer": "Something went wrong on my end. Please try again in a moment!",
            "source": "error"
        }


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Health check root endpoint."""
    return {
        "status": "Deepak's AI Assistant is live!",
        "version": "3.0",
        "model": "gemini-2.5-flash",
        "features": [
            "Keyword-based instant answers",
            "FAQ similarity matching",
            "Gemini AI fallback",
            "Voice-ready plain text responses",
            "CORS enabled for portfolio frontend"
        ]
    }


@app.get("/health")
async def health_check():
    """Simple health ping for uptime monitoring."""
    return {"status": "healthy", "model": "gemini-2.5-flash", "version": "3.0"}


@app.get("/keywords")
async def list_keywords():
    """Returns all keyword buckets loaded from profile.json — useful for debugging."""
    keyword_qa = profile_raw.get("keyword_qa", {})
    summary = {}
    for bucket, data in keyword_qa.items():
        summary[bucket] = {
            "keyword_count": len(data.get("keywords", [])),
            "answer_count":  len(data.get("answers", [])),
            "sample_keywords": data.get("keywords", [])[:5]
        }
    return {
        "total_buckets": len(summary),
        "buckets": summary
    }


@app.get("/faq")
async def list_faq():
    """Returns all FAQ questions loaded from profile.json — useful for debugging."""
    faq_list = profile_raw.get("faq", [])
    return {
        "total_faqs": len(faq_list),
        "questions": [item.get("question") for item in faq_list]
    }


@app.post("/test")
async def test_endpoint(request: Request):
    """
    Test endpoint — returns all three engine results for a question.
    Useful during development to see which engine would fire.
    """
    data = await request.json()
    question = data.get("question", "").strip()

    keyword_hit  = find_keyword_answer(question)
    faq_hit      = find_faq_answer(question)

    return {
        "question": question,
        "keyword_engine": keyword_hit or "no match",
        "faq_engine":     faq_hit     or "no match",
        "gemini_would_run": keyword_hit is None and faq_hit is None
    }
