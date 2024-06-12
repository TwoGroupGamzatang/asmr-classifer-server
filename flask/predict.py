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
import copy

keyword_to_label = {
    "웹": "웹 개발",
    "프론트엔드": "웹 개발",
    "모바일": "모바일 앱 개발",
    "앱": "모바일 앱 개발",
    "안드로이드": "모바일 앱 개발",
    "UI": "UI/UX디자인",
    "UX": "UI/UX디자인",
    "사용자 경험": "UI/UX디자인",
    "서버": "서버 개발",
    "시스템": "서버 개발",
    "DB": "DB 관리",
    "데이터베이스": "DB 관리",
    "SQL": "DB 관리",
    "아키텍쳐": "아키텍쳐",
    "클라우드": "아키텍쳐",
    "보안": "보안",
    "배포": "운영 배포",
    "운영": "운영 배포",
    "머신러닝": "머신러닝",
    "모델": "머신러닝",
    "AI": "머신러닝",
    "데이터과학": "데이터과학",
    "데이터": "데이터과학",
    "생성형": "생성형 AI",
    "genai": "생성형 AI",
    "genAI": "생성형 AI",
    "gpt": "생성형 AI",
    "GPT": "생성형 AI",
    "LLM": "생성형 AI",
    "추천": "추천 시스템",
    "프로젝트 계획": "프로젝트 계획",
    "계획 수립": "프로젝트 계획",
    "방법론": "프로젝트 방법론",
    "프로젝트 진행": "프로젝트 방법론",
    "agile": "프로젝트 방법론",
    "관리도구": "프로젝트 관리도구/기술",
    "관리 도구": "프로젝트 관리도구/기술",
    "관리 기술": "프로젝트 관리도구/기술",
    "프로젝트 관리": "프로젝트 관리도구/기술",
    "작업 관리": "프로젝트 관리도구/기술",
    "품질 관리": "품질 관리",
    "테스트": "품질 관리",
    "유지 보수": "품질 관리"
}

univ_label_mapping = {
    '웹 개발': 0,
    '모바일 앱 개발': 1,
    'UI/UX디자인': 2,
    '서버 개발': 3,
    'DB 관리': 4,
    '아키텍쳐': 5,
    '보안': 6,
    '운영 배포': 7,
    '머신러닝': 8,
    '데이터과학': 9,
    '생성형 AI': 10,
    '추천 시스템': 11,
    '프로젝트 계획': 12,
    '프로젝트 방법론': 13,
    '프로젝트 관리도구/기술': 14,
    '품질 관리': 15
}

univ_inverse_label_mapping = {v: k for k, v in univ_label_mapping.items()}


def boost_probabilities_based_on_keywords(text, predicted_probs, boost_factor=0.2):
    boosted_probs = copy.deepcopy(predicted_probs)
    applied_labels = set()  # 이미 적용된 라벨을 추적하기 위한 집합
    for keyword, label in keyword_to_label.items():
        if keyword in text and label not in applied_labels:
            if label in boosted_probs:
                boosted_probs[label] += boost_factor
                if boosted_probs[label] > 1.0:
                    boosted_probs[label] = 1.0
                applied_labels.add(label)  # 라벨이 한 번만 증가되도록 설정
    return boosted_probs


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
        probs = torch.sigmoid(outputs)


    
    predicted_probs = {univ_inverse_label_mapping[i]: probs[0][i].item() for i in range(len(probs[0]))}    
    boosted_probs = boost_probabilities_based_on_keywords(text, predicted_probs)   
    # 모든 클래스의 확률
    sorted_predicted_probs = dict(sorted(boosted_probs.items(), key=lambda item: item[1], reverse=True))
    print(sorted_predicted_probs)
        
    # 가장 높은 확률을 가진 클래스
    return sorted_predicted_probs