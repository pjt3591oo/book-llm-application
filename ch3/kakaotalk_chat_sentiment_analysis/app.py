from datasets import load_dataset
from transformers import pipeline

dataset = load_dataset('csv', data_files="kakaotalk_chat_sentiment_analysis/kakaochat.csv")
model_id = 'hun3359/klue-bert-base-sentiment'
model_pipeline = pipeline("text-classification", model=model_id)
''' 
필요없는 row 조건

User
    방장봇

Message
    삭제된 메시지입니다.
    사진
    이모티콘
    {USER}님이 나갔습니다.
    {USER}님이 들어왔습니다.
'''

def filter_row(row):
    user = row['User']
    
    if user == '방장봇':
        return None
    if row['Message'] in ['삭제된 메시지입니다.', '사진', '이모티콘', '님이 나갔습니다.', '님이 들어왔습니다.']:
        return None
    if row['Message'].startswith('http'):
        return None
    if f'{user}님이 나갔습니다.' == row['Message']:
        return None
    if f'{user}님이 들어왔습니다.' == row['Message']:
        return None
    
    return row

def add_sentiment(row):
    try:
        rst = model_pipeline(row['Message'])
        row.setdefault('sentiment', rst[0].get('label', ''))
        row.setdefault('rate', str(rst[0].get('score', 0)))
    except:
        row['sentiment'] = 'error'
        row['rate'] = '0'

    return row

print('before: ', dataset['train'])
did_filter_dataset = dataset.filter(filter_row)
did_filter_dataset['train'].to_csv('kakaotalk_chat_sentiment_analysis/kakaochat_filtered.csv')

# print('after: ', did_filter_dataset['train'])
# sentiment_dataset = did_filter_dataset.map(add_sentiment)
# sentiment_dataset['train'].to_csv('kakaotalk_chat_sentiment_analysis/kakaochat_sentiment.csv')
