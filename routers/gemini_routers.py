from fastapi import APIRouter, HTTPException, Depends, Request
import os
from langchain_google_genai import ChatGoogleGenerativeAI

router = APIRouter(
    prefix="/gemini",
)

#Contornar bug de variável de ambiente não está sendo reconhecida
gemini_api_key = os.getenv("GOOGLE_API_KEY")
if not gemini_api_key:
    os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_KEY')

gemini = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

@router.get("/translate")
def translate_text(totranslate: str):
    suffix = 'Translate from English to French: '
    gemini_response = gemini.invoke(suffix + totranslate)
    return {"texto_original": totranslate, "texto_traduzido": gemini_response.dict()['content']}
