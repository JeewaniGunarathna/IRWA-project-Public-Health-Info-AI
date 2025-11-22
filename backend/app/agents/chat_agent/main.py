# backend/chat_agent/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from .engine import get_chat_response

app = FastAPI(title="Chat Agent")

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

class ChatResponse(BaseModel):
    reply: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    reply = get_chat_response(body.message)
    return ChatResponse(reply=reply) 
