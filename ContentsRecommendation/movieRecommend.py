import requests

def fetch_movie_recommendation(emotion):
    token = "***REMOVED***"
    headers = {
        'Authorization': f'Bearer {token}',
        'accept': 'application/json'
    }

    genre_map = {'기쁨': '35'}  # Example: '기쁨' corresponds to Comedy genre
    url = (
        "https://api.themoviedb.org/3/discover/movie?"
        "include_adult=false&include_video=false&language=ko-KR&page=1"
        "&sort_by=popularity.desc&with_watch_providers=8"  # Adjust provider code as needed
        f"&with_genres={genre_map[emotion]}"
    )

    response = requests.get(url, headers=headers)
    # 응답 인코딩을 UTF-8로 설정
    response.encoding = 'utf-8'
    data = response.json()

    # 첫 번째 결과만 추출
    first_result = data['results'][0]

    # 감독 정보를 가져오기 위한 추가적인 요청
    movie_id = first_result['id']
    credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
    credits_response = requests.get(credits_url, headers=headers)
    credits_response.encoding = 'utf-8'  # 인코딩 설정
    credits_data = credits_response.json()

    # 감독 이름 추출
    directors = [crew['name'] for crew in credits_data['crew'] if crew['job'] == 'Director']

    # 필요한 정보만을 추출하여 새로운 JSON 객체 구성
    custom_data = {
        "name": first_result.get('title', 'No title provided'),
        "content": first_result.get('overview', 'No description provided'),
        "image": f"https://image.tmdb.org/t/p/w500{first_result.get('poster_path', '')}",
        "genre": ', '.join([genre['name'] for genre in first_result.get('genres', [])]),  # Genre needs to be mapped from IDs to names
        "director": ', '.join(directors) if directors else 'No director found'
    }

    return custom_data

# Example usage
emotion = '기쁨'
movie_data = fetch_movie_recommendation(emotion)
print(movie_data)
