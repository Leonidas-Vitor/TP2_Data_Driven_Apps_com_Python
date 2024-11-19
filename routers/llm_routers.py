from fastapi import APIRouter, HTTPException, Depends, Request
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
import torch
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import os

router = APIRouter(
    prefix="/llm",
)

class LLMResponseModel(BaseModel):
    answer: str
    confidence: str
    sources: str
    data_date: str

@router.get("/question", response_model=LLMResponseModel)
def llama(question: str, data):
    device = torch.device("cpu")

    pipe = pipeline("text-generation",
                    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    torch_dtype=torch.bfloat16,
                    device=device)
    message = [
        {"role": "system",
            "content": "You are a investiment advisor, and you have to answer the following question:"},
        {"role": "user",
            "content": question},
        {"role": "user",
            "content": f'''Answer the question above using the data below: {data}'''},
        {"role": "system",
            "content": 
            '''You should return the answer in json format, with the following structure: 
            {
            'answer': 'your answer here',
            'confidence': 'your confidence here',
            'source': 'your source here',
            'data_date': 'your data date here'
            }
            '''}
        ]

    prompt = pipe.tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=False
    )

    prediction = pipe(prompt,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.2, top_k=50, top_p=0.95)

    response = prediction[0]['generated_text']
    return response

