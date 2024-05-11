import os
import requests
import random
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('BOOK_API_TOKEN')

#국립중앙도서관 API 사용
def fetch_book_recommendation(emotion):
    topic_map = {
        '기쁨': 'humor',
        '슬픔': 'tragedy',
        '분노': 'anger management',
        '두려움': 'self help'
    }

    query = topic_map.get(emotion, 'fiction')  # 주제 매핑 또는 기본값
    url = f"https://www.nl.go.kr/seoji/SearchApi.do?cert_key={token}&result_style=json&page_no=1&page_size=10&start_publish_date=20220509&end_publish_date=20220509"
    response = requests.get(url)
    data = response.json()

    # 유효한 결과만 추출
    valid_books = [book for book in data.get('docs', []) if book.get('TITLE_URL')]

    if not valid_books:
        return "No valid books found."

    # 랜덤하게 하나의 유효한 결과 선택
    selected_book = random.choice(valid_books)

    custom_data = {
        "name": selected_book.get('TITLE', 'No title provided'),
        "content": selected_book.get('BOOK_INTRODUCTION_URL', 'No description provided'),
        "image": selected_book.get('TITLE_URL', 'No image provided'),
        "genre": selected_book.get('SUBJECT', 'No genre provided'),
        "author": selected_book.get('AUTHOR', 'No author provided')
    }

    return custom_data

# 사용 예
emotion = '기쁨'
book = fetch_book_recommendation(emotion)
print(f"책 추천: {book}")
