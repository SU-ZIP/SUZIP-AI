import json
from ContentsRecommendation import movieRecommend, musicRecommend, bookRecommend

def recommend_contents(emotion):
    movie = movieRecommend.fetch_movie_recommendation(emotion)
    music = musicRecommend.fetch_music_recommendation(emotion)
    book = bookRecommend.fetch_book_recommendation(emotion)
    
    recommendations = {
        "movie": movie,
        "music": music,
        "book": book
    }
    
    # 딕셔너리를 JSON 문자열로 변환
    recommendations_json = json.dumps(recommendations, ensure_ascii=False, indent=4)

    return recommendations_json
