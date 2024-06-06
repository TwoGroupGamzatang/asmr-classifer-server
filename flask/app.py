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

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Excel 파일 경로 설정
excel_file_path = 'sum_article_extend.xlsx'

# 라벨 인코더 로드 및 태그 매핑 생성
label_mapping = {
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

inverse_label_mapping = {v: k for k, v in label_mapping.items()}

# 모델 초기화 및 로드
num_classes = 16
model_save_path = 'univ_classifier_20240529_125400.pth'
universal_classify_model = ArticleClassifier(num_classes=num_classes)
universal_classify_model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
universal_classify_model.eval()

tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

def encode_label(tag):
    return label_mapping.get(tag, -1)  # -1은 없는 태그에 대한 디폴트 값입니다.

def decode_label(label):
    return inverse_label_mapping.get(label, 'Unknown')  # 'Unknown'은 없는 라벨에 대한 디폴트 값입니다.

# 사용자의 관심 분야에 따라 개인 분류기를 구성하는 함수
def create_personal_classifier(user_preferences, userID):
    train_data = load_training_data(user_preferences)
    model = train_save_personal_classifier(train_data,tokenizer, user_preferences, userID)
    # TODO
    # training_data를 사용하여 개인 분류기를 학습 코드 작성
    return

# 분류 예측  함수
import torch
import torch.nn.functional as F

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

# personal_classifier 가져오기
def get_personal_classifier(userID,num_classes):
    # TODO
    # DB와 연결해 personal classifier 가져오기 구현
    model_save_path = f'{userID}.pth'
    personal_classify_model = ArticleClassifier(num_classes=num_classes)
    personal_classify_model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    personal_classify_model.eval()

    return personal_classify_model

# 사용자 관심사 따라 개인 분류기를 구성 엔드포인트
@app.route('/create_personal_classifier', methods=['POST'])
def create_personal_classifier_endpoint():
    data = request.get_json()
    userID = data['userID']
    user_preferences = data['user_preferences']
    create_personal_classifier(userID, user_preferences)
    return jsonify({'message': 'Personal classifier built successfully.'})

# 사용자의 관심 분야에 따라 개인 분류기 예측 엔드포인트
@app.route('/classify_with_personal_classifier', methods=['POST'])
def predict_with_personal_classifier_endpoint():
    data = request.get_json()
    # TODO
    # 개인변 상태 관리
    userID = data['userID']
    text = data['summarized_text']

    
    #personal_label_mapping = {0: 3, 1: 5, 2: 8}
    #inverse_personal_label_mapping = {v: k for k, v in personal_label_mapping.items()}

    personal_classifiy_model = get_personal_classifier(userID,num_classes=3)
    predicted_class, top_classes, top_probs = predict_with_classifier(text,personal_classifiy_model,tokenizer)

    # 확률 차이를 계산하여 컷하는 로직 추가
    possible_classes = [top_classes[0]]  # 첫 번째 클래스
    
    # 두 번째 항목부터 반복하면서 차이를 계산
    for i in range(1, len(top_probs)):
        if (top_probs[i-1] - top_probs[i]) < 0.2:
            possible_classes.append(top_classes[i])
        else:
            break  # 차이가 20% 이상이면 종료

    predicted_class = decode_label(predicted_class)
    top_tags = [inverse_label_mapping[class_idx] for class_idx in top_classes]
    possible_tags = [inverse_label_mapping[class_idx] for class_idx in possible_classes]
    return jsonify({
        'predicted_class': predicted_class,
        'possible_classes': possible_tags
    })

# 범용 분류기 예측 엔드포인트
@app.route('/classify_with_universal_classifier', methods=['POST'])
def predict_with_universal_classifier_endpoint():
    data = request.get_json()
    text = data['summarized_text']
    predicted_class, top_classes, top_probs = predict_with_classifier(text, universal_classify_model, tokenizer)
    
    # 확률 차이를 계산하여 컷하는 로직 추가
    possible_classes = [top_classes[0]]  # 첫 번째 클래스
    
    # 두 번째 항목부터 반복하면서 차이를 계산
    for i in range(1, len(top_probs)):
        if (top_probs[i-1] - top_probs[i]) < 0.01:
            possible_classes.append(top_classes[i])
        else:
            break  # 차이가 20% 이상이면 종료
    
    predicted_class = decode_label(predicted_class)
    top_tags = [inverse_label_mapping[class_idx] for class_idx in top_classes]
    possible_tags = [inverse_label_mapping[class_idx] for class_idx in possible_classes]
    print(possible_tags)
    
    return jsonify({
        'predicted_class': predicted_class,
        'possible_classes': possible_tags
    })



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
