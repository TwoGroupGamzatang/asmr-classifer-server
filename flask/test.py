import pandas as pd
from utils import *

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

def encode_personal_label(tag, label_mapping):
    return label_mapping.get(tag, -1)  # 없는 태그는 -1로 인코딩

def load_training_data(userID):
    # TODO
    # DB에서 preference 해당하는 train_data 가져오기 구현 필요
    user_preferences,label_mapping = get_user_preferences(userID)
    article_df = pd.read_excel('sum_article_extend.xlsx')
    personal_article_df  = pd.DataFrame()
    personal_article_df = article_df[article_df['tag'].isin(user_preferences)]
    print(personal_article_df['tag'].value_counts())

    personal_article_df['tags_encoded'] = personal_article_df['tag'].apply(lambda x: encode_personal_label(x, label_mapping))
    summarized_text = personal_article_df['summarized']
    labels = personal_article_df['tags_encoded']
    print(personal_article_df['tags_encoded'].value_counts())

    return summarized_text, labels

user_preferences, label_mapping = get_user_preferences("user123")
print(user_preferences)
print(type(label_mapping))
load_training_data('user123')
