from fastapi import FastAPI, UploadFile, File, Body
from typing import List, Optional
import pandas as pd
import io

from .model import load_model, predict

app = FastAPI()

model, tokenizer = load_model()

demo_texts = [
    "Погода сегодня хорошая, солнечная и теплая.",
    "Научные открытия в области квантовой физики продолжаются.",
    "Экономика страны переживает кризис.",
    "Футбольная команда выиграла чемпионат."
]


@app.post("/predict-demo/")
def predict_demo_or_user(user_texts: Optional[List[str]] = Body(None)):
    texts_to_predict = user_texts if user_texts and user_texts != ['string'] else demo_texts
    predictions = predict(texts_to_predict, model, tokenizer)
    results = [{"text": text, "prediction": pred} for text, pred in zip(texts_to_predict, predictions)]
    return {"demo_results": results}


@app.post("/predict-from-csv/")
async def predict_from_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    texts = df['news'].tolist()
    predictions = predict(texts, model, tokenizer)
    results = [{"text": text, "prediction": pred} for text, pred in zip(texts, predictions)]
    return results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
