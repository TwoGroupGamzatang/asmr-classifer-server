from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from model import ArticleClassifier
from ArticleDataset import ArticleDataset
from train import train_save_personal_classifier, load_training_data
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss


def predict_with_classifier(text, model, tokenizer):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        input_ids = input_ids.to('cpu')
        attention_mask = attention_mask.to('cpu')
        outputs = model(input_ids, attention_mask)

        # 모델의 출력이 logits일 경우 그대로 사용
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        # 소프트맥스 함수로 확률 계산
        probabilities = F.softmax(logits, dim=1)
        
        # 모든 클래스의 확률
        all_probs = probabilities.cpu().numpy().flatten().tolist()
        all_classes = list(range(len(all_probs)))
        
        # 가장 높은 확률을 가진 클래스
        predicted_class = all_classes[all_probs.index(max(all_probs))]

    return predicted_class, all_classes, all_probs