# BiLSTMHubModel for Russian News Classification
![pic](https://raw.githubusercontent.com/data-silence/AIEdProjects/master/LSTM%20classifier/img/a_robotic_classifier_separates_newspapers_into_many_conveyor_streams__steampunk_style-c51d40fd-e31e-496c-8f2a-69db06af6ea6.webp)


This repository contains a BiLSTM model fine-tuned for Russian news classification. 
Model developed as part of a course [Рекуррентные сети в NLP и приложениях](https://stepik.org/course/188632/syllabus) on the Stepik platform.

The model is designed to classify Russian news articles into one of eleven categories using a bidirectional LSTM architecture on top of BERT embeddings.

```python
categories = [
    'climate', 'conflicts', 'culture', 'economy', 'gloss', 'health',
    'politics', 'science', 'society', 'sports', 'travel'
    ]
```

## Model Demo

You can sample the model's work right now at [Hagging Face Spaces](ttps://huggingface.co/spaces/data-silence/lstm-news-classifier) 


## Model Description

- **Model Architecture**: BiLSTM (Bidirectional LSTM) with BERT embeddings
- **Input**: Tokenized Russian text of news articles
- **Output**: Predicted category label for the news article (one of five classes)
- **Tokenizer**: BERT tokenizer (from Hugging Face Transformers library)
- **Embeddings**: BERT embeddings are used to represent the input text
- **Dropout**: 0.5 for regularization
- **Number of LSTM layers**: 2
- **Hidden size**: 256 (LSTM layer size)
- **Bidirectional**: True

## Dataset

The model was trained and evaluated on the "rus_news_classifier" dataset, which contains news articles in Russian along with their corresponding category labels. The dataset is available on the Hugging Face Hub: [data-silence/rus_news_classifier](https://huggingface.co/datasets/data-silence/rus_news_classifier).

You can find in the repository notebooks which contain the process of collecting data, and transforming it into a balanced dataset for classification tasks.

The full version of the original dataset and a description of its structure can be found [here](https://huggingface.co/datasets/data-silence/lenta.ru_2-extended)

## Usage

To use this model for predicting categories of Russian news articles, you can load it using the Hugging Face Transformers library:

```python
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


def get_predictions(news: str, model) -> str:
    with torch.no_grad():
        inputs = tokenizer(news, return_tensors="pt")
        del inputs['token_type_ids']
        output = model.forward(**inputs)
    id_best_label = torch.argmax(output[0, :], dim=-1).detach().cpu().numpy()
    prediction = categories[id_best_label]
    return prediction


# Использование классификатора
get_predictions('В Париже завершилась церемония завершения Олимпийский игр', model=model)
# 'sports'
```


## Training

The model was fine-tuned using a custom training loop with PyTorch and Hugging Face's Trainer class. The following hyperparameters were used:

    Batch size: 16
    Learning rate: 2e-5
    Number of epochs: 1
    Optimizer: AdamW

## Evaluation

The model was evaluated on the validation set of the "rus_news_classifier" dataset. The following metrics were used for evaluation:

    Loss: 0.6788
    Accuracy: 77.78%
    Precision: 78.93%
    Recall: 77.78%
    F1-score: 77.94%

Evaluation Details

    Model Preparation Time: 0.004 seconds
    Evaluation Runtime: 1324.91 seconds
    Samples per Second: 10.86
    Steps per Second: 0.679

These results suggest that the model performs well on the validation set, with a balanced performance across precision, recall, and F1-score.

```
@misc{data-silence2024,
  title={BiLSTMHubModel for Russian News Classification},
  author={data-silence},
  year={2024},
  howpublished={\url{https://huggingface.co/data-silence/rnn-news-classifier}},
}
```

## License

This model is licensed under the MIT License.