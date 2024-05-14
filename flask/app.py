from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer
import os
from model import ArticleClassifier 

# Flask 애플리케이션 설정
app = Flask(__name__)
CORS(app)

# 모델 초기화 및 로드
num_classes = 4     # 총 4개의 클래스로 분류
model_save_path = 'classifier_tmp.pth'  # 사전 학습된 분류 모델 경로
model = ArticleClassifier(num_classes=num_classes)  # 모델 어떻게 동작하는지 정의
model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))    # 사전 학습된 모델 불러오기
model.eval()

# 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")    # 문장 임베딩(벡터화)위한 KoBert

# 예측 함수
def predict(text, model, tokenizer):
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
        predicted_class = torch.argmax(outputs, dim=1).item()

    return predicted_class

# 예측 엔드포인트
@app.route('/predict', methods=['POST'])
def ajax():
    data = request.get_json()
    text = data['text']
    print(f"Received text: {text}")  # 요청이 도착 테스트
    predicted_class = predict(text, model, tokenizer)
    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
