import os
import requests
import random  # 랜덤 모듈 추가
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('MOVIE_API_TOKEN')

def fetch_movie_recommendation(emotion):
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json'
    }

    genre_map = {
        '기쁨': '35',
        '분노': '28',
        '슬픔': '18',
        '상처': '10751',
        '불안': '12'
    }
    
    url = (
        "https://api.themoviedb.org/3/discover/movie?"
        "include_adult=false&include_video=false&language=ko-KR&page=1"
        "&sort_by=popularity.desc&with_watch_providers=8"
        f"&with_genres={genre_map[emotion]}"
    )

    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'
    data = response.json()

    if 'results' in data and data['results']:
        # 랜덤하게 하나의 영화 선택
        random_movie = random.choice(data['results'])

        # 감독 정보를 가져오기 위한 추가적인 요청
        movie_id = random_movie['id']
        credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
        credits_response = requests.get(credits_url, headers=headers)
        credits_data = credits_response.json()

        # 감독 이름 추출
        directors = [crew['name'] for crew in credits_data['crew'] if crew['job'] == 'Director']

        # Fetching genres list
        genre_name_url = "https://api.themoviedb.org/3/genre/movie/list?api_key=YOUR_API_KEY&language=ko-KR"
        genre_response = requests.get(genre_name_url, headers=headers)
        genre_data = genre_response.json()

        genres = {genre['id']: genre['name'] for genre in genre_data['genres']} if 'genres' in genre_data else {}
        genre_ids = random_movie.get('genre_ids', [])
        first_genre_name = ', '.join(genres.get(g_id, "No genre found") for g_id in genre_ids)

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

# Example usage
emotion = '분노'
movie_data = fetch_movie_recommendation(emotion)
print(movie_data)
