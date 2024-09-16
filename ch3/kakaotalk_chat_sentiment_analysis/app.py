from datasets import load_dataset
from transformers import pipeline
import json

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
'''

def filter_row(row):
    if row['User'] == '방장봇':
        return None
    if row['Message'] in ['삭제된 메시지입니다.', '사진', '이모티콘']:
        return None
    return row

def add_sentiment(row):
    try:
        rst = model_pipeline(row['Message'])
        row['Sentiment'] = rst[0]['label']
    except:
        row['Sentiment'] = 'error'
    return row

print('before: ', dataset['train'])
did_filter_dataset = dataset.filter(filter_row)
did_filter_dataset['train'].to_csv('kakaotalk_chat_sentiment_analysis/kakaochat_filtered.csv')

print('after: ', did_filter_dataset['train'])
sentiment_dataset = did_filter_dataset.map(add_sentiment)
sentiment_dataset['train'].to_csv('kakaotalk_chat_sentiment_analysis/kakaochat_sentiment.csv')


# user_sentiment = {}
# count = 1
# dataset_size = len(dataset['train'])
# for row in dataset['train']:
#     try :
#         rst = model_pipeline(row['Message'])
#         print(f'============  {count} / {dataset_size}  ============')
#         # print('user: ', row['User'])
#         print('MSG : ', row['Message'])
#         print('result: ', rst)
#         user_sentiment.setdefault(row['User'], {})
#         user_sentiment[row['User']].setdefault(rst[0]['label'], 0)
#         user_sentiment[row['User']][rst[0]['label']] += 1
#         count += 1
#         print('========================')
#         print('')
#         print('')
#     except:
#         print('>>>>>> error: ', row['Message'])

# with open('./user_sentiment.json','w') as f:
#   json.dump(user_sentiment, f, ensure_ascii=False, indent=4)