import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import uvicorn
from vllm import LLM
import numpy as np
import torch
import argparse

# ====== 0. Initialize arguments ======

parser = argparse.ArgumentParser()
parser.add_argument("--emb_model_path", type=str, default="/YOUR_PATH/Qwen3-Embedding-0.6B")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()


# ====== 1. Initialize model ======
print("🚀 Loading Qwen3-Embedding-0.6B model, please wait..")
semantic_model = LLM(
    model=args.emb_model_path,
    task="embed",
    tensor_parallel_size=1,
    # gpu_memory_utilization=0.1,
)
print("✅ Model loaded.")



# ====== 2. Define API request and response models ======
class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]


# ====== 3. Create API ======
app = FastAPI(title="Qwen3 Embedding API", version="1.0")


# ====== 4. API endpoints ======
@app.post("/embed", response_model=EmbedResponse)
def get_embeddings(request: EmbedRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="Invalid input")

    try:
        outputs = semantic_model.embed(request.texts, use_tqdm=False)
        embeddings = [o.outputs.embedding for o in outputs]
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====== 5. Start API server ======
if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)

