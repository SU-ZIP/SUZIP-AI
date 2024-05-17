import os
import requests
import random
import sys
from dotenv import load_dotenv

# 콘솔 인코딩 설정 (Windows에서 필요할 수 있음)
sys.stdout.reconfigure(encoding='utf-8')

# .env 파일 로드
load_dotenv()
token = os.getenv('MOVIE_API_TOKEN')

def fetch_movie_recommendation(emotion):
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json'
    }

    genre_map = {
        '기쁨': '35',  # 코미디
        '분노': '28',  # 액션
        '슬픔': '18',  # 드라마
        '상처': '10751',  # 가족
        '불안': '12'  # 모험
    }

    # 랜덤 페이지 번호 설정
    random_page = random.randint(1, 100)
    
    url = (
        f"https://api.themoviedb.org/3/discover/movie?"
        f"include_adult=false&include_video=false&language=ko&page={random_page}"
        f"&sort_by=popularity.desc&with_watch_providers=8"
        f"&with_genres={genre_map[emotion]}"
    )

    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'  # UTF-8 인코딩 설정
    data = response.json()

    if 'results' in data and data['results']:
        valid_movies = [movie for movie in data['results'] if movie.get('overview')]
        
        if not valid_movies:
            return "No valid movies found"

        random_movie = random.choice(valid_movies)
        movie_id = random_movie['id']
        credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
        credits_response = requests.get(credits_url, headers=headers)
        credits_response.encoding = 'utf-8'  # UTF-8 인코딩 설정
        credits_data = credits_response.json()

        directors = [crew['name'] for crew in credits_data['crew'] if crew['job'] == 'Director']

        # 장르 리스트 가져오기
        genre_name_url = f"https://api.themoviedb.org/3/genre/movie/list?language=ko-KR"
        genre_response = requests.get(genre_name_url, headers=headers, params={'api_key': token})
        genre_response.encoding = 'utf-8'  # UTF-8 인코딩 설정
        genre_data = genre_response.json()

        if 'genres' in genre_data:
            genres = {genre['id']: genre['name'] for genre in genre_data['genres']}
        else:
            genres = {}

        genre_ids = random_movie.get('genre_ids', [])
        first_genre_name = genres.get(genre_ids[0], "No genre found") if genre_ids else "No genre provided"

        custom_data = {
            "name": random_movie.get('title', 'No title provided'),
            "content": random_movie.get('overview', 'No description provided'),
            "image": f"https://image.tmdb.org/t/p/w500{random_movie.get('poster_path', '')}",
            "genre": first_genre_name,
            "director": ', '.join(directors) if directors else 'No director found'
        }

        return custom_data
    else:
        print("No 'results' key or empty results in data:", data)  # Debug output
        return "No results found"
