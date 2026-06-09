from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import json
import os

app = FastAPI()

# Allow your frontend to talk to the backend without security blocks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to Google Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

# Load your resume data safely
try:
    with open("profile.json", "r") as f:
        profile_data = json.dumps(json.load(f))
except Exception:
    profile_data = "No resume data found."

@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
        user_question = data.get("question", "")

        # Strict rules for the AI so it never returns an empty response
        prompt = f"""
        You are the personal AI assistant for DEEPAK L.
        Here is his resume data: {profile_data}
        
        Answer the user's question conversationally, as if you are Deepak's AI representative.
        Keep the answer to 1 or 2 short sentences so it is easy to speak out loud.
        
        CRITICAL RULE: You MUST answer every single question with text. If the user asks something that is not in the resume, simply reply exactly with: "I'm sorry, I only have information about Deepak's professional background and resume."
        Do not use bold text, asterisks, or markdown. Use plain conversational text only.
        
        User Question: {user_question}
        """

        response = model.generate_content(prompt)
        
        # Safely extract text to prevent server crashes
        if response and hasattr(response, 'parts') and response.parts:
            return {"answer": response.text.replace('*', '')} # Removes weird AI formatting
        else:
            return {"answer": "I'm sorry, my AI brain couldn't process that question."}
            
    except Exception as e:
        print(f"Backend Error: {e}")
        # If the API crashes, send a safe voice response instead of a server error
        return {"answer": "I can only answer questions related to Deepak's professional experience."}
