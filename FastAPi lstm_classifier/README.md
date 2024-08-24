# News Category Classification Model

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68.0+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0+-red.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.10.0+-yellow.svg)
![Docker](https://img.shields.io/badge/Docker-20.10.0+-blue.svg)

This repository contains a FastAPI application that uses a BiLSTM classifier model to predict categories for news articles. The model is trained to classify news into one of the following categories:

- climate
- conflicts
- culture
- economy
- gloss
- health
- politics
- science
- society
- sports
- travel

## Model Description

The model is a Bidirectional LSTM (BiLSTM) classifier that uses a pre-trained BERT model for initial text embeddings. It's designed to understand the context and content of news articles and assign them to the most appropriate category.
All detailed information about the model, its training and features, and the dataset on which the model was trained can be found [here](https://github.com/data-silence/AIEdProjects/tree/master/LSTM%20classifier).

The model itself and checkpoints to it you can find on [HF](https://huggingface.co/data-silence/lstm-news-classifier), there you can also try the [demo](https://huggingface.co/spaces/data-silence/lstm-news-classifier) version of the model.

## Architecture:
1. BERT layer for initial text embedding
2. Bidirectional LSTM layer for sequence processing
3. Fully connected layer for final classification

## Usage

The API provides three endpoints for making predictions:

1. `/predict-from-csv/`: Accepts a CSV file containing news articles and returns predictions for each article.
2. `/predict-demo/` (GET): Returns predictions for a set of demo news articles.
3. `/predict-demo/` (POST): Accepts a list of news articles and returns predictions for each.

## Example usage with curl:

```bash
# Predict demo articles
curl -X GET http://localhost:8000/predict-demo/

# Predict custom articles
curl -X POST http://localhost:8000/predict-demo/ \
     -H "Content-Type: application/json" \
     -d '{"user_texts": ["Новый космический телескоп запущен на орбиту", "Цены на нефть достигли рекордного уровня"]}'

# Predict from CSV file
curl -X POST http://localhost:8000/predict-from-csv/ \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/news_articles.csv"
```

## Installation and Running

Clone this repository
Install dependencies: pip install -r requirements.txt
Run the FastAPI application: uvicorn app.main:app --reload

Alternatively, you can use Docker:
```bash
docker-compose build
docker-compose up
```
The API will be available at `http://localhost:8000.`

## API Documentation
Once the application is running, you can access the automatic API documentation at `http://localhost:8000/docs.`

## Detailed API Usage Examples
Here are more detailed examples of how to interact with the API endpoints using Python and the requests library.
1. Predicting Categories for Demo Articles

```python
import requests

# GET request to predict demo articles
response = requests.get("http://localhost:8000/predict-demo/")
if response.status_code == 200:
    results = response.json()
    for item in results["demo_results"]:
        print(f"Text: {item['text']}")
        print(f"Predicted Category: {item['prediction']}")
        print("---")
else:
    print("Error:", response.status_code, response.text)
```
2. Predicting Categories for Custom Articles
```python
import requests
import json

# Custom news articles
custom_articles = [
    "Новый вид рыбы обнаружен в Марианской впадине",
    "Правительство объявило о снижении налогов для малого бизнеса",
    "Ученые разработали новый метод лечения рака"
]

# POST request to predict custom articles
response = requests.post(
    "http://localhost:8000/predict-demo/",
    data=json.dumps({"user_texts": custom_articles}),
    headers={"Content-Type": "application/json"}
)

if response.status_code == 200:
    results = response.json()
    for item in results["demo_results"]:
        print(f"Text: {item['text']}")
        print(f"Predicted Category: {item['prediction']}")
        print("---")
else:
    print("Error:", response.status_code, response.text)
```

3. Predicting Categories from a CSV File

First, create a CSV file named `news_articles.csv` with the following content:
```csv
news
"Глобальное потепление ускоряется, показывают новые исследования"
"Переговоры о мире на Ближнем Востоке зашли в тупик"
"Новая выставка современного искусства открылась в Лувре"
```

Then use the following Python code to send the file to the API:
```python
import requests

# POST request to predict from CSV file
with open("news_articles.csv", "rb") as file:
    response = requests.post(
        "http://localhost:8000/predict-from-csv/",
        files={"file": file}
    )

if response.status_code == 200:
    results = response.json()
    for item in results:
        print(f"Text: {item['text']}")
        print(f"Predicted Category: {item['prediction']}")
        print("---")
else:
    print("Error:", response.status_code, response.text)
```

These examples demonstrate how to interact with each endpoint of the API using Python. 
They cover predicting categories for demo articles, custom articles, and articles from a CSV file. 
Users can adapt these examples to their specific needs and integrate them into their applications.