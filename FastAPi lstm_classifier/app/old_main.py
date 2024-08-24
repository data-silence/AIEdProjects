from fastapi import FastAPI, Body, UploadFile, File
from typing import List, Optional

from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

import pandas as pd
import io

import torch
import torch.nn as nn
from transformers import BertModel

import sys
sys.modules['__main__'] = sys.modules[__name__]

# Здесь определение класса модели BiLSTMClassifier
class BiLSTMClassifier(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_layers, dropout):
        super(BiLSTMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim, num_layers=n_layers,
                            bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            embedded = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        lstm_out, _ = self.lstm(embedded)
        pooled = torch.mean(lstm_out, dim=1)
        logits = self.fc(self.dropout(pooled))

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}  # Возвращаем словарь
        return logits  # Возвращаем логиты, если метки не переданы


# torch.serialization.add_safe_globals([BiLSTMClassifier])

# словарь для расшифровки категорий, т.к. модель предскажет только номер класса
categories = ['climate', 'conflicts', 'culture', 'economy', 'gloss', 'health',
              'politics', 'science', 'society', 'sports', 'travel']




# загружаем модель с HF
repo_id = "data-silence/lstm-news-classifier"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model_path = hf_hub_download(repo_id=repo_id, filename="model.pth")


model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)  # weights_only=True
model.eval()


# получаем предсказания модели
def get_predictions(news: str, model) -> str:
    with torch.no_grad():
        inputs = tokenizer(news, return_tensors="pt")
        del inputs['token_type_ids']
        output = model.forward(**inputs)
    id_best_label = torch.argmax(output[0, :], dim=-1).detach().cpu().numpy()
    prediction = categories[id_best_label]
    return prediction


app = FastAPI()


# Функция для предсказания категории текста
def predict(texts: list[str]) -> list[str]:
    predictions = []
    for text in texts:
        pred = get_predictions(text, model)
        predictions.append(pred)
    return predictions


# Endpoint для предсказания на основе загруженного CSV файла
@app.post("/predict-from-csv/")
async def predict_from_csv(file: UploadFile = File(...)):
    # Чтение CSV файла
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    # Предполагаем, что в файле есть колонка 'news'
    texts = df['news'].tolist()

    # Получение предсказаний
    predictions = predict(texts)

    # Добавление предсказаний в DataFrame и преобразование в CSV
    # df['predictions'] = predictions
    # result = df.to_csv(index=False)

    results = [{"text": text, "prediction": pred} for text, pred in zip(demo_texts, predictions)]


    # return {"predictions_csv": result}
    return {"results": results}

# Демонстрационные данные для второго эндпоинта
demo_texts = [
    "Погода сегодня хорошая, солнечная и теплая.",
    "Научные открытия в области квантовой физики продолжаются.",
    "Экономика страны переживает кризис.",
    "Футбольная команда выиграла чемпионат."
]


# Endpoint для демонстрационного применения модели
@app.get("/predict-demo/")
def predict_demo():
    # Получение предсказаний на демо-данных
    predictions = predict(demo_texts)

    # Возвращаем результат в виде JSON
    results = [{"text": text, "prediction": pred} for text, pred in zip(demo_texts, predictions)]

    return {"demo_results": results}


@app.post("/predict-demo/")
def predict_demo(user_texts: Optional[List[str]] = Body(None)):
    # Если пользователь не предоставил тексты или предоставил пустой список, используем демо-тексты
    texts_to_predict = user_texts if user_texts != ['string'] else demo_texts

    # Получение предсказаний
    predictions = predict(texts_to_predict)

    # Формируем результат
    results = [{"text": text, "prediction": pred} for text, pred in zip(texts_to_predict, predictions)]

    return {"demo_results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
