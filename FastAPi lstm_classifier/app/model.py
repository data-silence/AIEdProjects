import sys
import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer
from huggingface_hub import hf_hub_download
from typing import List, Tuple

# Обеспечиваем, чтобы модуль был доступен при загрузке модели
sys.modules['__main__'] = sys.modules[__name__]


# Здесь определяется класс модели BiLSTMClassifier, без чего модули HF не смогут понять параметры модели
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
            return {"loss": loss, "logits": logits}
        return logits


def load_model() -> Tuple[BiLSTMClassifier, AutoTokenizer]:
    # Определяем параметры загрузки модели
    repo_id = "data-silence/lstm-news-classifier"
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model_path = hf_hub_download(repo_id=repo_id, filename="model.pth")

    # Загружаем модель
    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()

    return model, tokenizer


categories = ['climate', 'conflicts', 'culture', 'economy', 'gloss', 'health',
              'politics', 'science', 'society', 'sports', 'travel']


def predict(texts: List[str], model: BiLSTMClassifier, tokenizer: AutoTokenizer) -> List[str]:
    predictions = []
    for text in texts:
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            del inputs['token_type_ids']
            output = model(**inputs)
        id_best_label = torch.argmax(output, dim=1).item()
        prediction = categories[id_best_label]
        predictions.append(prediction)
    return predictions
