from pymongo import MongoClient, errors

def insert_user_preferences(userId, preferences):
    url = "mongodb+srv://inwoo920621:pasly0920@asmr.nxdtmlt.mongodb.net/scraper?retryWrites=true&w=majority"
    """
    :param userId: 사용자 ID (문자열)
    :param preferences: 사용자 선호도 (문자열 리스트)
    """
    client = MongoClient(url)
    db = client.scraper
    collection = db.preference

    # 중복 검사: 이미 해당 userId가 존재하는지 확인
    existing_user = collection.find_one({"userId": userId})
    if existing_user:
        print(f"User with userId {userId} already exists.")
        return {"message": "오류. 이미 해당 user가 존재합니다.", "status": "duplicate"}

    label_mapping = {pref: idx for idx, pref in enumerate(preferences)}
    document = {
        "userId": userId,
        "preferences": preferences,
        "preferencesMapping": label_mapping
    }

    try:
        collection.insert_one(document)
        print(f"User preferences for {userId} inserted successfully.")
        return {"message": "관심사 생성완료", "status": "success"}
    except errors.PyMongoError as e:
        print(f"Error inserting user preferences: {e}")
        return {"message": "관심사 생성 실패", "status": "error"}

    collection.insert_one(document)

def update_user_preferences(userId, new_preferences):
    url = "mongodb+srv://inwoo920621:pasly0920@asmr.nxdtmlt.mongodb.net/scraper?retryWrites=true&w=majority"
    """
    :param user_id: 사용자 ID (문자열)
    :param new_preferences: 새로운 사용자 선호도 (문자열 리스트)
    """
    client = MongoClient(url)
    db = client.scraper
    collection = db.preference

    new_label_mapping = {pref: idx for idx, pref in enumerate(new_preferences)}

    # 업데이트할 문서와 업데이트 내용 설정
    query = {"userId": userId}
    new_values = {
        "$set": {
            "preferences": new_preferences,
            "preferencesMapping": new_label_mapping
        }
    }
    # 문서 업데이트
    result = collection.update_one(query, new_values)

    if result.matched_count > 0:
        print("관심사 업데이트.")
    else:
        print("error. cannot find userId")

def get_user_preferences(userId):
    url = "mongodb+srv://inwoo920621:pasly0920@asmr.nxdtmlt.mongodb.net/scraper?retryWrites=true&w=majority"
    """
    :param user_id: 사용자 ID (문자열)
    :param connection_string: MongoDB 연결 문자열
    :return: 사용자 선호도 (문자열 리스트) 또는 None
    """
    client = MongoClient(url)
    db = client.scraper
    collection = db.preference

    # 사용자 ID로 문서 찾기
    query = {"userId": userId}
    document = collection.find_one(query)

    if document:
        user_preferences = document.get("preferences", None)
        label_mapping = document.get("preferencesMapping", None)
        return user_preferences, label_mapping
    else:
        print("해당 사용자 ID를 가진 문서를 찾을 수 없습니다.")
        return None