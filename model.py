from fastapi import FastAPI, HTTPException
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer
import torch
import torch.nn as nn
import gluonnlp as nlp
from torch.utils.data import Dataset, DataLoader
import numpy as np
import boto3

app = FastAPI()

# AWS S3 설정
s3_client = boto3.client(
    's3',
    aws_access_key_id='YOUR_AWS_ACCESS_KEY',
    aws_secret_access_key='YOUR_AWS_SECRET_KEY'
)

# 토크나이저 로드
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

# 모델 정의
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=6, dr_rate=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
        self.dr_rate = dr_rate

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(), attention_mask=attention_mask.float().to(token_ids.device), return_dict=False)
        if self.dr_rate:
            pooler = self.dropout(pooler)
        return self.classifier(pooler)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(BertModel.from_pretrained('skt/kobert-base-v1'), dr_rate=0.5).to(device)
model.load_state_dict(torch.load('/content/drive/My Drive/Model/BERTClassifier.pt', map_location=device))
model.eval()

@app.get("/interpret")
async def interpret_text(file_key: str, bucket: str):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=file_key)
        text = response['Body'].read().decode('utf-8')
        result = predict(text)
        return {"original_text": text, "interpreted_text": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def predict(predict_sentence):
    data = [predict_sentence, '0']  # 레이블은 예측 시 필요하지 않으므로 0으로 설정
    dataset_another = [data]
    
    another_test = BERTDataset(dataset_another, 0, 1, tokenizer, max_len, True, False)
    test_dataloader = DataLoader(another_test, batch_size=1, num_workers=0)
    
    model.eval()
    with torch.no_grad():
        for token_ids, valid_length, segment_ids, _ in test_dataloader:
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            out = model(token_ids, valid_length, segment_ids)
            logits = out.detach().cpu().numpy()
            emotion_id = np.argmax(logits, axis=1)[0]  # 가장 높은 확률을 가진 클래스 인덱스
            
    emotions = ["불안", "슬픔", "당황", "기쁨", "분노", "상처"]
    return emotions[emotion_id]


vicorn.run(app, host="0.0.0.0", port=8000))
