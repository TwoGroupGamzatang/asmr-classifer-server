# model.py
import torch
import torch.nn as nn
from transformers import BertModel

class ArticleClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(ArticleClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("klue/bert-base")
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.pooler_output
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        probs = torch.sigmoid(logits)
        return probs