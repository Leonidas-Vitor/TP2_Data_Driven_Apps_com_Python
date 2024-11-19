from fastapi import FastAPI, HTTPException, Depends
from routers.gpt2_routers import router as gpt2_router
from routers.nlp_routers import router as nlp_router
from routers.fake_routers import router as fake_router
from routers.gemini_routers import router as gemini_router
import pandas as pd

api = FastAPI()

#api.state.stockPrices = stockPrices

api.include_router(gpt2_router)
api.include_router(nlp_router)
api.include_router(fake_router)
api.include_router(gemini_router)


@api.get("/")
def read_root():
    return {"message": "API rodando!"}

