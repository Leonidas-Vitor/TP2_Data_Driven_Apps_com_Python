from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from typing import Optional
from langchain_community.llms import FakeListLLM


router = APIRouter(
    prefix="/fake_llm",
)

fake_llm = FakeListLLM(responses=[
        "Olá!",
        "Tudo e com você?",
        "Como posso ajudar você hoje?",
        "Não entendi, poderia reformular?",
        "Minhas respostas são limitadas, mas podemos conversar sobre qualquer coisa!",
        ])

@router.get("/chat")
def chat_with_fake_llm(msg : Optional[str] = None):
    response = fake_llm.invoke(msg.lower(), inputs=1)
    return {"response": response}



@router.get("/")
def status():
    return "Welcome to the Fake LLM API!"
    