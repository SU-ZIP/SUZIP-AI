from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from kobert_tokenizer import KoBERTTokenizer
import numpy as np
from torch.utils.data import DataLoader

from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np

#from recommendContents import recommend_contents

import json
from ContentsRecommendation import movieRecommend, musicRecommend, bookRecommend
from gpt4o import generate_sentence

app = FastAPI()

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,   #히든 레이어 사이즈, BERT base 768
                 num_classes = 6,     # 최종 분류할 감정수
                 dr_rate = None,      # 드롭아웃 비율
                 params = None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes) #선형 레이어
        if dr_rate:
            self.dropout = nn.Dropout(p = dr_rate)

    def gen_attention_mask(self, token_ids, valid_length): #토큰 길이 정보 기반 어텐션 마스크
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):  #순전파 정의 (입력토큰, 토큰유효길이, 세그먼트ID)
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device),return_dict = False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
    


# 사전 훈련된 BERT 모델과 토크나이저 초기화
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
#vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
tok = tokenizer.tokenize


# 토크나이저 및 모델 초기화
model = BERTClassifier(bertmodel, dr_rate=0.5)
model_path = 'model-walesmin.pt'  # 모델 파일 경로
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

class Item(BaseModel):
    sentence: str

# 파라미터 설정
max_len = 512         #입력 텍스트 최대길이

# 데이터셋 토큰화
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]
    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False)
    test_dataloader = DataLoader(another_test, batch_size=1, num_workers=0)

    model.eval()

    emotions = ["불안", "슬픔", "기쁨", "분노", "상처"]
    results = []
    for token_ids, valid_length, segment_ids, label in test_dataloader:
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)
        logits = out.detach().cpu().numpy()
        predicted_emotion = emotions[np.argmax(logits)]
        results.append(predicted_emotion)

    return results[0] if results else "감정 분석 결과를 찾을 수 없습니다."

def recommend_contents(emotion):
    movie = movieRecommend.fetch_movie_recommendation(emotion)
    music = musicRecommend.fetch_music_recommendation(emotion)
    book = bookRecommend.fetch_book_recommendation()
    
    recommendations = {
        "movie": movie,
        "music": music,
        "book": book
    }
    
    # 딕셔너리를 JSON 문자열로 변환
    recommendations_json = json.dumps(recommendations, ensure_ascii=False, indent=4)
    return recommendations_json

@app.post("/predict/")
async def predict_sentiment(item: Item):
    try:
        #result = predict(item.sentence)
        #result = recommend_contents(predict(item.sentence))
        #return {"emotion": result}

        emotion_result = predict(item.sentence)
        recommendations_json = recommend_contents(emotion_result)
        content_recommendation = json.loads(recommendations_json)  # JSON 문자열을 딕셔너리로 변환
        sentence = generate_sentence(emotion_result)

        return {
            "emotion": emotion_result,
            "recommendations": content_recommendation,
            "sentence": sentence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"Hello": "World"}
