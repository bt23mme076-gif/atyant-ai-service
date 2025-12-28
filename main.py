from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import time

app = FastAPI()

# 🚀 OFFICIAL WAY: Library khud URL manage karegi
HF_TOKEN = "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"
# Model wahi hai jo hum use kar rahe the
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Client initialize karein
client = InferenceClient(api_key=HF_TOKEN)

class TextData(BaseModel):
    text: str

@app.post("/embed")
async def get_embedding(data: TextData):
    try:
        # 🚀 Feature Extraction call (Ye embeddings generate karta hai)
        # Isme '404 Not Found' aane ka chance 0% hai
        vector = client.feature_extraction(data.text, model=MODEL_ID)
        
        # Convert to list (Hugging Face Hub results are often numpy-like)
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        
        # 🚀 CLEANING: Agar nested list [[...]] hai toh flat karke [...] bhejta hai
        if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
            vector = vector[0]
            
        print("✅ Vector generated successfully using Official Library!")
        return {"embedding": vector}

    except Exception as e:
        # Agar model load ho raha ho (timeout error)
        if "503" in str(e) or "loading" in str(e).lower():
            print("⏳ Model is loading... wait a few seconds and try again.")
            return {"error": "Model is loading", "details": "Please retry in 10-15 seconds"}
        
        print(f"❌ Python Error: {str(e)}")
        return {"error": str(e)}

@app.get("/")
def home():
    return {"message": "Atyant AI Service (Official Hub Library) is Live!"}