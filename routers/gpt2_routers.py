from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
import pandas as pd
import os
from transformers import pipeline, set_seed

router = APIRouter(
    prefix="/gpt2",
)

generator = pipeline("text-generation", model="gpt2")
set_seed(42)

class InputText(BaseModel):
    text: str

@router.get("/generate")
async def generate_text(input_data: InputText = Depends()):
    try:
        output = generator(input_data.text, max_length=100, num_return_sequences=1, truncation=True
                           ,temperature=0.7, do_sample=True)
        return {"input": input_data.text, "generated_text": output[0]["generated_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar texto: {str(e)}")

@router.get("/")
def root():
    return {"message": "GPT-2 Text Generator API is running!"}