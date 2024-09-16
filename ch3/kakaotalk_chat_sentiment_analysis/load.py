from datasets import load_dataset
import json
dataset = load_dataset('csv', data_files="kakaotalk_chat_sentiment_analysis/kakaochat_sentiment.csv")

user_sentiment = {}

for row in dataset['train']:
    # print(row)
    user_sentiment.setdefault(row['User'], {})
    user_sentiment[row['User']].setdefault(row['Sentiment'], 0)
    user_sentiment[row['User']][row['Sentiment']] += 1

print(user_sentiment['멍개'])
# with open('user_sentiment.json','w') as f:
#   json.dump(user_sentiment, f, ensure_ascii=False, indent=4)