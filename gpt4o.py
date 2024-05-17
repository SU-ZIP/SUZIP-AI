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
        {"role": "user", "content": f"나는 일기장에 쓸 문장을 원해. 감정에 따라 따뜻하고 친근한 스타일의 문장을 한국어로 작성해줘 또한 감정에 대해 사용자에게 도움이 되는 말을 해줘. 인용구는 넣지 말아줘고 한문장으로 하되 20자 안으로 해줘. 감정 은 {emotion}"}  
   ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
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
