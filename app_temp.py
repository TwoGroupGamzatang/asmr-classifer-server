from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer
import pandas as pd
from model import ArticleClassifier

app = Flask(__name__)
CORS(app)

# Excel 파일 경로 설정
excel_file_path = 'sum_article_extend.xlsx'

# 모델 초기화 및 로드
num_classes = 16
model_save_path = 'univ_classifier_tmp.pth'
universal_classify_model = ArticleClassifier(num_classes=num_classes)
universal_classify_model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
universal_classify_model.eval()

# 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

def load_training_data(user_preferences):
    #TODO
    # DB에서 preference 해당하는 train_data 가져오기 구현 필요
    pass

# 사용자의 관심 분야에 따라 개인 분류기를 구성하는 함수
def create_personal_classifier(user_preferences):
    train_data = load_training_data(user_preferences)
    # TODO
    # training_data를 사용하여 개인 분류기를 학습 코드 작성
    pass

# 개인 분류기 예측 함수
def predict_with_personal_classifier(text, model, tokenizer):
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

# 범용 분류기 예측 함수
def predict_with_universal_classifier(text, model, tokenizer):
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

# personal_classifier 가져오기
def get_personal_classifier(userID):
    # TODO
    # DB와 연결해 personal classifier 가져오기 구현
    pass

# 사용자 관심사와 persoanl classifier 업데이트 함수
def update_user_preferences(user_preferences):
    # TODO
    # DB와 연결해 관심사 업데이트 필요
    pass

# 사용자 관심사 따라 개인 분류기를 구성 엔드포인트
@app.route('/create_personal_classifier', methods=['POST'])
def build_personal_classifier_endpoint():
    data = request.get_json()
    userID = data['userID']
    user_preferences = data['user_preferences']
    create_personal_classifier(userID, user_preferences)
    return jsonify({'message': 'Personal classifier built successfully.'})

# 사용자의 관심 분야에 따라 개인 분류기 예측 엔드포인트
@app.route('/classify_with_personal_classifier', methods=['POST'])
def predict_with_personal_classifier_endpoint():
    data = request.get_json()
    userID = data['userID']
    text = data['summarized_text']
    personal_classifiy_model = get_personal_classifier(userID)
    predicted_class = predict_with_personal_classifier(text,personal_classifiy_model,tokenizer)
    return jsonify({'predicted_class': predicted_class})

# 범용 분류기 예측 엔드포인트
@app.route('/classify_with_universal_classifier', methods=['POST'])
def predict_with_universal_classifier_endpoint():
    data = request.get_json()
    text = data['summarized_text']
    predicted_class = predict_with_universal_classifier(text,universal_classify_model,tokenizer)
    return jsonify({'predicted_class': predicted_class})

# 사용자의 관심 분야를 업데이트하고 개인 분류기를 업데이트하는 엔드포인트
@app.route('/update_personal_classifier', methods=['POST'])
def update_user_preferences_endpoint():
    data = request.get_json()
    userID = data['userID']
    user_preferences = data['user_preferences']
    update_user_preferences(userID, user_preferences)
    return jsonify({'message': 'User preferences updated successfully.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)