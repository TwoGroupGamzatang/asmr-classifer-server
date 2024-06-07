from pymongo import MongoClient

def insert_user_preferences(user_id, preferences):
    url = "mongodb+srv://inwoo920621:pasly0920@asmr.nxdtmlt.mongodb.net/scraper?retryWrites=true&w=majority"
    """
    :param user_id: 사용자 ID (문자열)
    :param preferences: 사용자 선호도 (문자열 리스트)
    """
    client = MongoClient(url)
    db = client.scraper
    collection = db.preference

    label_mapping = {pref: idx for idx, pref in enumerate(preferences)}
    document = {
        "userID": user_id,
        "user_preferences": preferences,
        "user_preferences_mapping": label_mapping
    }

    collection.insert_one(document)

def update_user_preferences(user_id, new_preferences):
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
    query = {"userID": user_id}
    new_values = {
        "$set": {
            "user_preferences": new_preferences,
            "user_preferences_mapping": new_label_mapping
        }
    }
    # 문서 업데이트
    result = collection.update_one(query, new_values)

    if result.matched_count > 0:
        print("관심사 업데이트.")
    else:
        print("error. cannot find userID")

def get_user_preferences(user_id):
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
    query = {"userID": user_id}
    document = collection.find_one(query)

    if document:
        user_preferences = document.get("user_preferences", None)
        label_mapping = document.get("user_preferences_mapping", None)
        return user_preferences, label_mapping
    else:
        print("해당 사용자 ID를 가진 문서를 찾을 수 없습니다.")
        return None