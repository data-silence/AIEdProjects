import gradio as gr
import torch
import torch.nn as nn
from transformers import BertModel
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

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


categories = ['climate', 'conflicts', 'culture', 'economy', 'gloss', 'health',
              'politics', 'science', 'society', 'sports', 'travel']

repo_id = "data-silence/lstm-news-classifier"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model_path = hf_hub_download(repo_id=repo_id, filename="model.pth")

model = torch.load(model_path)


def predict(news: str) -> str:
    with torch.no_grad():
        inputs = tokenizer(news, return_tensors="pt")
        del inputs['token_type_ids']
        output = model.forward(**inputs)
    id_best_label = torch.argmax(output[0, :], dim=-1).detach().cpu().numpy()
    prediction = categories[id_best_label]
    return prediction


# Создание интерфейса Gradio
iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, label="Enter news text | Введите текст новости"),
    outputs=[
        gr.Label(label="Predicted category | Предсказанная категория"),
        gr.Label(label="Category probabilities | Вероятности категорий")
    ],
    title="News Classifier | Классификатор новостей",
    description="Enter the news text in any language and the model will predict its category. | Введите текст новости на любом языке, и модель предскажет её категорию"
)

iface.launch()