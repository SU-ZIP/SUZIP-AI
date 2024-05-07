import requests

#Google Books API에 요청을 보내는 코드
def fetch_book_recommendation(emotion):
    topic_map = {
        '기쁨': 'humor',
        '슬픔': 'tragedy',
        '분노': 'anger management',
        '두려움': 'self help'
    }

    # Google Books API 사용
    query = topic_map.get(emotion, 'fiction')  # 주제 매핑 또는 기본값
    url = f"https://www.googleapis.com/books/v1/volumes?q=subject:{query}"
    response = requests.get(url)
    data = response.json()

    # 첫 번째 책 결과만 추출
    first_result = data['items'][0]['volumeInfo']

    # 필요한 정보만을 추출하여 새로운 JSON 객체 구성
    custom_data = {
        "name": first_result.get('title', 'No title provided'),
        "content": first_result.get('description', 'No description provided'),
        "image": first_result.get('imageLinks', {}).get('thumbnail', 'No image provided'),
        "genre": ', '.join(first_result.get('categories', ['No genre provided'])),  # 장르 배열
        "author": ', '.join(first_result.get('authors', ['No author provided']))  # 저자 배열
    }

    return custom_data

# 사용 예
emotion = '기쁨'
book = fetch_book_recommendation(emotion)

print(f"책 추천: {book}")
