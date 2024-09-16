from datasets import load_dataset


klue_tc_train = load_dataset('klue', 'ynat', split='train')
klue_tc_eval = load_dataset('klue', 'ynat', split='validation')

print(klue_tc_train)
print(klue_tc_eval)

print(klue_tc_train[0])
print(klue_tc_train.features['label'].names)


# 불필요한 컬럼 제거
klue_tc_train = klue_tc_train.remove_columns(['guid', 'url', 'date'])
klue_tc_eval = klue_tc_eval.remove_columns(['guid', 'url', 'date'])

klue_tc_label = klue_tc_train.features['label']
print(klue_tc_label)
print(klue_tc_train.features['label'].int2str(1))

print(klue_tc_train.features['label'].int2str(klue_tc_train[0]['label']))
klue_tc_train[0]['label_str'] = klue_tc_train.features['label'].int2str(klue_tc_train[0]['label'])

def make_str_label (batch) :
    batch['label_str'] = klue_tc_label.int2str(batch['label'])


# 컬럼추가
klue_tc_train = klue_tc_train.map(make_str_label, batched=True, batch_size=1000)
print(klue_tc_train[0])


# 학습 / 검증 / 테스트 데이터셋 분할
train_dataset = klue_tc_train.train_test_split(test_size=10000, shuffle=True, seed=42)["test"]
print(train_dataset)
dataset = klue_tc_eval.train_test_split(test_size=1000, shuffle=True, seed=42)

test_dataset=dataset["test"]
print(test_dataset)

valid_dataset=dataset['train'].train_test_split(test_size=1000, shuffle=True, seed=42)['test']
print(valid_dataset)