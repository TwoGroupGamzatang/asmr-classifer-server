# model.py
import torch
import torch.nn as nn
from transformers import BertModel

class ArticleClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ArticleClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("monologg/kobert")
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.pooler_output
        logits = self.classifier(sequence_output)
        return logits
