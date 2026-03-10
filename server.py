from model import generate_text
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/generate")
def generate(text: str, num_chars: int, temperature: float):
    return {"generated_text": generate_text(text,num_chars,temperature)}