import requests

def fetch_movie_recommendation(emotion):
    token = "***REMOVED***"
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json'
    }

    genre_map = {'기쁨': '35'}  # Example: '기쁨' corresponds to Comedy genre
    
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
        first_result = data['results'][0]
        
        # 감독 정보를 가져오기 위한 추가적인 요청
        movie_id = first_result['id']
        credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
        credits_response = requests.get(credits_url, headers=headers)
        credits_data = credits_response.json()

        # 감독 이름 추출
        directors = [crew['name'] for crew in credits_data['crew'] if crew['job'] == 'Director']


        # Fetching genres list
        genre_name_url = "https://api.themoviedb.org/3/genre/movie/list?api_key=YOUR_API_KEY&language=ko-KR"
        genre_response = requests.get(genre_name_url, headers=headers)
        genre_data = genre_response.json()

        if 'genres' in genre_data:
            genres = {genre['id']: genre['name'] for genre in genre_data['genres']}
            genre_ids = first_result.get('genre_ids', [])
            first_genre_name = genres.get(genre_ids[0], "No genre found") if genre_ids else "No genre provided"
        else:
            print("No 'genres' key in genre_data:", genre_data)  # Debug output
            first_genre_name = "No genre provided"

        custom_data = {
            "name": first_result.get('title', 'No title provided'),
            "content": first_result.get('overview', 'No description provided'),
            "image": f"https://image.tmdb.org/t/p/w500{first_result.get('poster_path', '')}",
            "genre": first_genre_name,
            "director": ', '.join(directors) if directors else 'No director found'
        }

        return custom_data
    else:
        print("No 'results' key or empty results in data:", data)  # Debug output
        return "No results found"

# Example usage
emotion = '기쁨'
movie_data = fetch_movie_recommendation(emotion)
print(movie_data)
