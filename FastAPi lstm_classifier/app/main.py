from fastapi import FastAPI, UploadFile, File, Body
from typing import List, Optional
import pandas as pd
import io

from app.model import load_model, predict
from app.schemas import PredictionResult
from app.utils import demo_texts

app = FastAPI()

model, tokenizer = load_model()

@app.post("/predict-from-csv/", response_model=List[PredictionResult])
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Predict categories for news articles from a CSV file.

    :param file: Uploaded CSV file containing news articles.
    :return: List of prediction results.
    """
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    texts = df['news'].tolist()
    predictions = predict(texts, model, tokenizer)
    return [PredictionResult(text=text, prediction=pred) for text, pred in zip(texts, predictions)]

@app.get("/predict-demo/", response_model=List[PredictionResult])
def predict_demo():
    """
    Predict categories for demo news articles.

    :return: List of prediction results for demo texts.
    """
    predictions = predict(demo_texts, model, tokenizer)
    return [PredictionResult(text=text, prediction=pred) for text, pred in zip(demo_texts, predictions)]

@app.post("/predict-demo/", response_model=List[PredictionResult])
def predict_demo_or_user(user_texts: Optional[List[str]] = Body(None)):
    """
    Predict categories for user-provided news articles or demo texts.

    :param user_texts: Optional list of user-provided news articles.
    :return: List of prediction results.
    """
    texts_to_predict = user_texts if user_texts and user_texts != ['string'] else demo_texts
    predictions = predict(texts_to_predict, model, tokenizer)
    return [PredictionResult(text=text, prediction=pred) for text, pred in zip(texts_to_predict, predictions)]