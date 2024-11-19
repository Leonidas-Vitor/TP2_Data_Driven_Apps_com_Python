from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from fastapi import APIRouter, HTTPException, Depends, Request

router = APIRouter(
    prefix="/nlp",
)

model_fr = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
tokenizer_fr = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

model_de = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
tokenizer_de = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
translation_pipeline = pipeline("translation", model=model_de, tokenizer=tokenizer_de)
de_llm = HuggingFacePipeline(pipeline=translation_pipeline)


class TranslationRequest(BaseModel):
    text: str
    language: str

@router.get("/translate")
async def translate_text(totranslate: TranslationRequest = Depends()):
    if totranslate.language not in ["fr", "de"]:
        raise HTTPException(status_code=400, detail="Linguagem de destino não suportada. Por favor, escolha 'fr' ou 'de'.")
    else:
        translated_text = ''
        try:
            if totranslate.language == "fr":
                model = model_fr
                tokenizer = tokenizer_fr
                inputs = tokenizer(totranslate.text, return_tensors="pt", padding=True)
                translated = model.generate(**inputs)
                translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            else:
                translated_text = de_llm.invoke(totranslate.text)
            
            return {"texto_original": totranslate.text, "language": totranslate.language ,"texto_traduzido": translated_text}
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao traduzir o texto: {str(e)}")

@router.get("/")
async def root():
    return {"message": "English to French Or English to German Translation"}

