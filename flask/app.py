from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from utils import insert_user_preferences, update_user_preferences
from model import ArticleClassifier
from ArticleDataset import ArticleDataset
from predict import predict_with_classifier
from train import train_save_personal_classifier, load_training_data, load_personal_training_data
from utils import *
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from datetime import datetime

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 라벨 인코더 로드 및 태그 매핑 생성
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

# 모델 초기화 및 로드
num_classes = 16
model_save_path = 'univ_classifier_20240612_081304.pth'
universal_classify_model = ArticleClassifier(num_classes=num_classes)
universal_classify_model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
universal_classify_model.eval()

tokenizer = BertTokenizer.from_pretrained("monologg/kobert")


def univ_encode_label(tag):
    return univ_label_mapping.get(tag, -1)  # -1은 없는 태그에 대한 디폴트 값.

def univ_decode_label(label):
    return univ_inverse_label_mapping.get(label, 'Unknown')  # 'Unknown'은 없는 라벨에 대한 디폴트 값.

def encode_personal_label(tag, label_mapping):
    return label_mapping.get(tag, -1)  # 없는 태그는 -1로 인코딩   

# 사용자의 관심 분야에 따라 개인 분류기를 구성하는 함수
def create_personal_classifier(userID, user_preferences):
    summarized_text, labels = load_training_data(userID)
    train_save_personal_classifier(summarized_text, labels, tokenizer, user_preferences, userID)

    data_save_path = os.path.join("user_data", f'{userID}.json')
    personal_article_df = pd.DataFrame({
        'summarized_text': summarized_text,
        'labels': labels
    })
    personal_article_df.to_json(data_save_path, orient='records', lines=True)

    return


# personal_classifier 가져오기
def get_personal_classifier(userID):
    # TODO
    # DB와 연결해 personal classifier 가져오기 구현
    model_save_path = f'{userID}.pth'
    user_preferences, _ = get_user_preferences(userID)
    num_classes = len(user_preferences)
    personal_classify_model = ArticleClassifier(num_classes=num_classes)
    personal_classify_model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    personal_classify_model.eval()

    return personal_classify_model

# 사용자 관심사 따라 개인 분류기를 구성 엔드포인트
@app.route('/create_personal_classifier', methods=['POST'])
def create_personal_classifier_endpoint():
    data = request.get_json()
    userID = data['userId']
    print(userID)
    user_preferences = data['preferences']
    print(user_preferences)
    insert_user_preferences(userID,user_preferences)
    create_personal_classifier(userID, user_preferences)
    return jsonify({'message': '개인 분류기 생성 완료. user123.pth파일 생성된것 확인'})

# 사용자의 관심 분야에 따라 개인 분류기 예측 엔드포인트
@app.route('/classify_with_personal_classifier', methods=['POST'])
def predict_with_personal_classifier_endpoint():
    data = request.get_json()
    # TODO
    # 개인별 상태 관리
    userID = data['userId']
    text = data['summarized_text']
    #personal_label_mapping = {0: 3, 1: 5, 2: 8}
    #inverse_personal_label_mapping = {v: k for k, v in personal_label_mapping.items()}
    personal_classifiy_model = get_personal_classifier(userID)
    predicted_class_wth_probs = predict_with_classifier(text, personal_classifiy_model, tokenizer)
    print(predicted_class_wth_probs)

    # 확률 차이를 계산하여 컷하는 로직 추가
    possible_classes_all = [cls for cls, prob in predicted_class_wth_probs.items() if prob > 0.55][:5]
    predicted_class = possible_classes_all[0] if possible_classes_all else "기타"
    possible_classes = possible_classes_all[1:]
    
    
    
    return jsonify({
        'predicted_class': predicted_class,
        'possible_classes': possible_classes
    })

# 범용 분류기 예측 엔드포인트
@app.route('/classify_with_universal_classifier', methods=['POST'])
def predict_with_universal_classifier_endpoint():
    data = request.get_json()
    text = data['summarized_text']
    predicted_class_wth_probs = predict_with_classifier(text, universal_classify_model, tokenizer)
    
    # 확률 차이를 계산하여 컷하는 로직 추가
    possible_classes_all = [cls for cls, prob in predicted_class_wth_probs.items() if prob > 0.60][:5]
    predicted_class = possible_classes_all[0] if possible_classes_all else "기타"
    possible_classes = possible_classes_all[1:]  

    return jsonify({
        'predicted_class': predicted_class,
        'possible_classes': possible_classes
    })

# 사용자 classifier 업데이트하는 엔드포인트
@app.route('/update_personal_classifier', methods=['POST'])
def update_personal_classifier_endpoint():
    data = request.get_json()
    userID = data['userId']
    summarized_text = data['summarized_text']
    label = data['update_label']
    
    user_preference, label_mapping = get_user_preferences(userID)
    org_summarized_text, org_labels = load_personal_training_data(userID)
    
    new_summarized_text = pd.Series([summarized_text])
    new_labels = pd.Series([[label]])  # label을 리스트로 감싸서 전달
    new_labels = new_labels.apply(lambda x: [encode_personal_label(x[0], label_mapping)])  # 내부 값을 인코딩 후 다시 리스트로 감싸기

    merged_summarized_text = pd.concat([org_summarized_text, new_summarized_text], ignore_index=True)
    merged_labels = pd.concat([org_labels, new_labels], ignore_index=True)

    personal_data_path = os.path.join("user_data", f'{userID}.json')
    personal_article_df = pd.DataFrame({
        'summarized_text': merged_summarized_text,
        'labels': merged_labels
    })

    if os.path.exists(personal_data_path):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join("user_data", f'{userID}_{current_time}.json')
        os.rename(personal_data_path, backup_path)
    personal_article_df.to_json(personal_data_path, orient='records', lines=True)
    
    # Get the last element of merged_labels
    last_label = merged_labels.iloc[-1]

    # Print the last label and its type
    print(f"Last label: {last_label}")
    print(f"Type of last label: {type(last_label)}")
    print(merged_labels)
    print(merged_summarized_text)
    
    train_save_personal_classifier(merged_summarized_text, merged_labels, tokenizer, user_preference, userID)

    return jsonify({'message': f'개인 분류기 업데이트됨. user{userID}.pth(최신파일), user{userID}.pth_2024XXXX (예전 파일)'})

# 사용자의 관심 분야를 업데이트
@app.route('/update_user_preference', methods=['POST'])
def update_user_preferences_endpoint():
    data = request.get_json()
    userID = data['userId']
    user_preferences = data['preferences']
    update_user_preferences(userID, user_preferences)
    return jsonify({'message': '관심사 업데이트 됨. DB를 확인해주세요'})

# 사용자 개인 분류기가 있는지 확인 
@app.route('/check_personal_classifier', methods=['POST'])
def check_personal_classfier_endpoint():
    data = request.get_json()
    userID = data['userId']

    classifier_file = f'{userID}.pth'
    if os.path.isfile(classifier_file):
        return jsonify({"exists": True})
    else:
        return jsonify({"exists": False})
    

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=5000)

