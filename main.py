from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

# 🚀 STEP 1: Load environment variables correctly
load_dotenv() 

app = FastAPI()

# 🚀 STEP 2: Initialize Hugging Face Client
# 'HF_TOKEN' aapki .env file se aayega
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

if not HF_TOKEN:
    print("❌ WARNING: HF_TOKEN not found in environment variables!")

client = InferenceClient(api_key=HF_TOKEN)

class TextData(BaseModel):
    text: str

@app.post("/embed")
async def get_embedding(data: TextData):
    try:
        # 🚀 Generate Embeddings
        vector = client.feature_extraction(data.text, model=MODEL_ID)
        
        # Convert to list if it's a numpy-like object
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        
        # 🚀 CLEANING: Ensure it's a flat list [...] not [[...]]
        if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
            vector = vector[0]
            
        print(f"✅ Vector generated for: {data.text[:30]}...")
        return {"embedding": vector}

    except Exception as e:
        error_msg = str(e)
        # Handling Model Loading State (503)
        if "503" in error_msg or "loading" in error_msg.lower():
            print("⏳ Model is loading on Hugging Face...")
            return {"error": "Model is loading", "details": "Please retry in 10 seconds"}
        
        # Handling Unauthorized (401)
        if "401" in error_msg:
            print("❌ Token Error: Unauthorized. Check your HF_TOKEN.")
            return {"error": "Unauthorized", "details": "Invalid or expired Hugging Face token"}

        print(f"❌ AI Error: {error_msg}")
        return {"error": error_msg}

@app.get("/")
def home():
    return {"message": "Atyant AI Service is Online 🚀"}