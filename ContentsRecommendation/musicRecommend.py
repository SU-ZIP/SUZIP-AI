import requests
import random
import os
from dotenv import load_dotenv

load_dotenv()  # 환경 변수 불러오기

client_id = os.getenv('SPOTIFY_CLIENT_ID')
client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')

def get_spotify_token():
    url = "https://accounts.spotify.com/api/token"
    response = requests.post(url, {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    })
    return response.json()['access_token']

def fetch_music_recommendation(emotion):
    genre_map = {
        '기쁨': ['happy', 'joy', 'exciting K-POP'],
        '분노': ['angry', 'hard', 'upset'],
        '슬픔': ['sad', 'gloomy', 'dark K-POP'],
        '상처': ['hurt', 'sick'],
        '불안': ['anxious']
    }

    token = get_spotify_token()
    headers = {'Authorization': f'Bearer {token}'}

    # 감정에 맞는 키워드로 플레이리스트 검색, 배열 내 요소를 OR 연산자로 결합
    keywords = genre_map.get(emotion, ['music'])
    query = " OR ".join(keywords)
    url = f"https://api.spotify.com/v1/search?q={query}&type=playlist&limit=1"
    response = requests.get(url, headers=headers)
    playlists = response.json().get('playlists', {}).get('items', [])
    
    if not playlists:
        return "No playlists found."

    # 첫 번째 플레이리스트 선택
    playlist_id = playlists[0]['id']
    
    # 플레이리스트의 트랙 가져오기
    tracks_url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    tracks_response = requests.get(tracks_url, headers=headers)
    tracks = tracks_response.json().get('items', [])
    
    if not tracks:
        return "No tracks found in the playlist."
    
    # 랜덤 트랙 선택
    track = random.choice(tracks)['track']
    artists_names = ', '.join(artist['name'] for artist in track['artists'])

    custom_data = {
        "name": track['name'],
        "image": track['album']['images'][0]['url'],
        "artist": artists_names
    }

    return custom_data

# 감정 입력 및 음악 추천 실행
emotion = '기쁨'
recommended_track = fetch_music_recommendation(emotion)

print(recommended_track)
