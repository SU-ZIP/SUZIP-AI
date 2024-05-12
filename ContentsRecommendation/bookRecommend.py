import os
import requests
import random
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('BOOK_API_TOKEN')

def fetch_book_recommendation(attempt=1, max_attempts=5):
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

    allowed_subjects = {'3', '7', '8'}
    random_page = random.randint(1, 100)

    url = f"https://www.nl.go.kr/seoji/SearchApi.do?cert_key={token}&result_style=json&page_no={random_page}&page_size=10&start_publish_date=20200509&end_publish_date=20220509"
    response = requests.get(url)
    data = response.json()

    valid_books = [
        book for book in data.get('docs', [])
        if book.get('TITLE_URL') and str(book.get('SUBJECT')) in allowed_subjects
    ]

    if valid_books:
        selected_book = random.choice(valid_books)
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
    
    else:
        if attempt < max_attempts:
            return fetch_book_recommendation(attempt + 1, max_attempts)
        else:
            return "No valid books found after multiple attempts."
