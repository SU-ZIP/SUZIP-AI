import requests


#Spotify API 인증을 위해 액세스 토큰을 얻는 함수
def get_spotify_token(client_id, client_secret):
    auth_url = "https://accounts.spotify.com/api/token"

    auth_response = requests.post(auth_url, {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
    })

    auth_response_data = auth_response.json()

    return auth_response_data['access_token']


#감정에 따른 플레이리스트 검색 (예시)
def fetch_music_recommendation(emotion, token):
    playlist_map = {
        '기쁨': 'Happy Hits',
        '슬픔': 'Sad Songs',
        '분노': 'Rock Hard',
        '두려움': 'Relax'
    }

    #플레이리스트 이름으로 Spotify에서 검색
    search_query = playlist_map.get(emotion, 'Pop')
    url = f"https://api.spotify.com/v1/search?q={search_query}&type=playlist&limit=1"
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
    }
    response = requests.get(url, headers=headers)
    data = response.json()

    #첫 번째 플레이리스트의 첫 번째 트랙 추출
    playlist_id = data['playlists']['items'][0]['id']
    playlist_tracks_url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks?limit=1"
    tracks_response = requests.get(playlist_tracks_url, headers=headers)
    tracks_data = tracks_response.json()

    first_track = tracks_data['items'][0]['track']

    #필요한 정보만을 추출하여 새로운 JSON 객체 구성
    custom_data = {
        "name": first_track['name'],
        "content": first_track['album']['name'],  # 앨범 이름을 '설명'으로 사용
        "image": first_track['album']['images'][0]['url'],  # 첫 번째 이미지 URL
        "genre": ', '.join([genre['name'] for genre in first_track.get('genres', [])]),  # 장르 정보가 없는 경우 처리
        "artist": ', '.join([artist['name'] for artist in first_track['artists']])  # 아티스트 이름 목록
    }

    return custom_data


#클라이언트 ID와 시크릿을 사용하여 토큰 가져오기
client_id = '***REMOVED***'
client_secret = '***REMOVED***'
token = get_spotify_token(client_id, client_secret)


emotion = '기쁨'
music_data = fetch_music_recommendation(emotion, token)


print(music_data)
