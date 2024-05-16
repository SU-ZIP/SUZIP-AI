import os
import openai
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

# 감정에 따른 문장 생성 함수
def generate_sentence(emotion):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"You will be provided with emotion, and your task is to write a sentence for me today in disney style chat. emotion includes happy, anger, sadness, hurt, anxiety. I want sentence in Korean. i dont want a word 디즈니 in the sentence. {emotion}. "}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1,
        max_tokens=100,
        # top_p=1,
        # frequency_penalty=0,
        # presence_penalty=0
    )
    
    sentence = response['choices'][0]['message']['content'].strip()
    return sentence

# 사용자로부터 감정을 입력받음
# user_emotion = input("감정을 입력해주세요 ")

# # 해당 감정에 대한 문장 생성
# sentence = generate_sentence(user_emotion)
# print(sentence)
