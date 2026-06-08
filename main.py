from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("profile.json", "r") as file:
    profile_data = json.load(file)

@app.post("/ask")
async def ask_ai(request: Request):
    data = await request.json()
    user_question = data.get("question")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    system_prompt = f"""
    You are the voice assistant for {profile_data['name']}'s web portfolio. 
    Use this JSON data to answer the user's question: {json.dumps(profile_data)}
    
    Rules:
    - Keep answers under 3 sentences so it sounds natural when spoken aloud.
    - Be professional but conversational.
    
    User Question: {user_question}
    """
    
    response = model.generate_content(system_prompt)
    return {"answer": response.text}
