import os
import requests
import random
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('BOOK_API_TOKEN')

#국립중앙도서관 API 사용
def fetch_book_recommendation(emotion):

    #분류 매핑
    subject_map = {
        '0': '총류',
        '1': '철학',
        '2': '종교',
        '3': '사회과학',
        '4': '자연과학',
        '5': '기술과학',
        '6': '예술',
        '7': '언어',
        '8': '문학',
        '9': '역사',
    }
    
    # 랜덤 페이지 번호 생성 (예: 1에서 10 사이)
    random_page = random.randint(1, 10)

    url = f"https://www.nl.go.kr/seoji/SearchApi.do?cert_key={token}&result_style=json&page_no={random_page}&page_size=10&start_publish_date=20200509&end_publish_date=20220509"
    response = requests.get(url)
    data = response.json()

    # 이미지가 존재하는 결과만 추출
    valid_books = [book for book in data.get('docs', []) if book.get('TITLE_URL')]

    if not valid_books:
        return "No valid books found."

    # 랜덤하게 하나의 유효한 결과 선택
    selected_book = random.choice(valid_books)
    
    # 저자 이름 파싱
    author_text = selected_book.get('AUTHOR', 'No author provided')
    author = author_text.split(':')[-1].strip() if ':' in author_text else author_text.strip()

    subject_description = subject_map.get(selected_book.get('SUBJECT', ''), 'No subject provided')

    custom_data = {
        "name": selected_book.get('TITLE', 'No title provided'),
        "image": selected_book.get('TITLE_URL', 'No image provided'),
        "genre": subject_description,
        "author": author
    }

    return custom_data

# 사용 예
emotion = '기쁨'
book = fetch_book_recommendation(emotion)
print(f"책 추천: {book}")