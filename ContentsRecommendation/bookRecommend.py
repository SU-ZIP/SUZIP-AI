import requests
import lxml.etree as ET
import random
import json
import sys
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()
token = os.getenv('BOOK_API_TOKEN')

# 콘솔 인코딩 설정 (Windows에서 필요할 수 있음)
sys.stdout.reconfigure(encoding='utf-8')

# 최근 선택된 항목을 추적하는 리스트
recent_items = []

# 감정과 분류 매핑
emotion_to_genre_code = {
    "기쁨": "6",   # 인문과학
    "슬픔": "11",    # 문학
    "분노": "5",   # 사회과학
    "불안": "4", # 자연과학
    "상처": None    # 상처: 모든 장르에서 선택
}

# XML 파일 다운로드 및 캐싱
cache_file = "cached_response.xml"
url = "https://nl.go.kr/NL/search/openApi/saseoApi.do"
params = {
    "key": token,
    "startRowNumApi": "1",
    "endRowNumApi": "250",
    "start_date": "20200101",
    "end_date": "20240501"
}

def fetch_and_cache_xml():
    response = requests.get(url, params=params)
    response.encoding = 'utf-8'
    with open(cache_file, "wb") as f:
        f.write(response.content)
    return response.content

def load_xml():
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return f.read()
    else:
        return fetch_and_cache_xml()

# XML 데이터 로드
def load_items():
    xml_content = load_xml()
    root = ET.fromstring(xml_content)
    return root.findall(".//item")

items = load_items()

# 랜덤한 item 요소 선택 함수
def get_random_item_by_genre(genre_code, exclude_items):
    if genre_code:
        available_items = [item for item in items if item not in exclude_items and item.find("drCode") is not None and item.find("drCode").text == genre_code and item.find("drCodeName") is not None and item.find("drCodeName").text]
    else:
        available_items = [item for item in items if item not in exclude_items and item.find("drCodeName") is not None and item.find("drCodeName").text]
    if not available_items:
        return None
    return random.choice(available_items)

# 감정을 받아서 JSON 결과 반환 함수
def fetch_book_recommendation(emotion):
    global recent_items

    # 감정에 해당하는 장르 코드 찾기
    genre_code = emotion_to_genre_code.get(emotion)

    # 랜덤한 item 요소 선택
    random_item = get_random_item_by_genre(genre_code, recent_items)

    if random_item is not None:
        # 최근 선택된 항목 업데이트 (최대 5개까지 추적)
        recent_items.append(random_item)
        if len(recent_items) > 10:
            recent_items.pop(0)
        
        # 출력할 태그 목록 및 대체 이름
        tags_to_print = {
            "drCodeName": "genre",
            "recomtitle": "name",
            "recomauthor": "author",
            "recomfilepath": "image"
        }
        
        # 선택한 태그와 값을 저장할 딕셔너리
        selected_item_data = {}
        
        # item 요소의 자식 요소들 중 선택한 태그만 딕셔너리에 저장
        for child in random_item:
            if child.tag in tags_to_print:
                # 태그 이름을 대체 이름으로 변경하여 저장
                selected_item_data[tags_to_print[child.tag]] = child.text
        
        # JSON 형식으로 변환하여 반환
        return selected_item_data
    else:
        return {"error": "No unique item found"}