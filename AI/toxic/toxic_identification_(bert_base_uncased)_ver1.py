#pip install torch transformers datasets scikit-learn


import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict


name = 'toxic_(bert_base_uncased)_ver1_3차'

##################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv('./Data_Analysis/Data/toxic_sentence_merged.csv')

# 훈련 데이터 & 테스트 데이터 분할 (80:20)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data["sentence"].tolist(), data["label"].tolist(), test_size=0.2, random_state=42,stratify=data["label"].tolist()
)
# 사전 훈련된 BERT 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 텍스트를 토큰화 (BERT 입력 형태로 변환)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# DatasetDict 생성 (Hugging Face Datasets 사용)
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

# 토큰화 적용
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

def convert_labels_to_float(dataset):
    dataset = dataset.map(lambda x: {"labels": float(x["label"])})
    return dataset

# 훈련 및 테스트 데이터셋 변환
dataset["train"] = convert_labels_to_float(dataset["train"])
dataset["test"] = convert_labels_to_float(dataset["test"])
##################################################################################################
##################################################################################################
##################################################################################################

# 사전 훈련된 BERT 모델 불러오기 (이진 분류용)
output_dir = f"./Data_Analysis/Model/{name}/ckp/"
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
model.to(device)  # GPU 사용

# 학습 파라미터 설정
training_args = TrainingArguments(
    output_dir= output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Trainer 설정 (Hugging Face Trainer API 사용)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# 모델 학습 시작
trainer.train()


###########################################################################
###########################################################################
###########################################################################

import os
save_dir = f"./Data_Analysis/Model/{name}/"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

###########################################################################
###########################################################################
###########################################################################

from transformers import BertForSequenceClassification, BertTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # 현재 사용 중인 디바이스 확인
save_dir = f"./Data_Analysis/Model/{name}/"
model = BertForSequenceClassification.from_pretrained(save_dir).to(device)
tokenizer = BertTokenizer.from_pretrained(save_dir)


##################################################################################################
##################################################################################################
##################################################################################################
def predict(text):
    model.eval()  # 평가 모드 설정
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits  # 모델 예측값
        prob = torch.sigmoid(logits).item()  # 확률 변환
    return prob  # 독소 조항일 확률 (0~1 사이 값)

# 테스트 데이터
test_data = [
    ["을은 갑의 요청이 있을 경우, 정해진 계약 기간과 관계없이 추가적인 납품을 진행해야 한다.", 1],
    ["갑은 을의 재고 상황과 관계없이 주문량을 자유롭게 조정할 수 있으며, 을은 이에 무조건 응해야 한다.", 1],
    ["을은 갑의 판매 전략에 따라 원가 이하의 가격으로 납품해야 하며, 이에 대한 손실 보전을 요구할 수 없다.", 1],
    ["본 계약 체결 이후에도 갑은 을의 유통망을 직접 통제할 수 있으며, 을은 이를 거부할 수 없다.", 1],
    ["을은 갑의 경영 전략에 따라 가격 및 판매 정책을 조정해야 하며, 이에 대한 협의 권한이 없다.", 1],
    ["갑은 을의 납품 기한을 사전 협의 없이 조정할 수 있으며, 을은 이에 즉시 응해야 한다.", 1],
    ["을은 갑의 판매 촉진을 위해 추가적인 제품을 무상으로 제공해야 하며, 이에 대한 대가를 요구할 수 없다.", 1],
    ["본 계약의 종료 여부는 갑이 단독으로 결정하며, 을은 이에 대해 어떠한 권리도 주장할 수 없다.", 1],
    ["갑은 을의 생산 과정에 개입할 권리를 가지며, 을은 이에 대해 거부할 수 없다.", 1],
    ["을은 계약이 종료된 후에도 일정 기간 동안 갑이 요청하는 조건을 유지하여 제품을 공급해야 한다.", 1],
    ["계약 당사자는 계약의 이행을 위해 상호 협력하며, 문제 발생 시 협의를 통해 해결해야 한다.", 0],
    ["을은 계약된 일정에 따라 제품을 납품하며, 일정 변경이 필요한 경우 사전에 협의한다.", 0],
    ["본 계약의 조항은 양측의 동의 없이 일방적으로 변경될 수 없다.", 0],
    ["계약 해지 시, 당사자는 합의된 절차에 따라 서면으로 통보해야 한다.", 0],
    ["갑은 을의 정당한 사유 없이 계약 조건을 일방적으로 변경할 수 없다.", 0],
    ["을은 계약 이행 중 발생하는 문제를 갑에게 즉시 보고하고 협의해야 한다.", 0],
    ["본 계약은 계약서에 명시된 기한 동안 적용되며, 연장은 양측 협의를 통해 진행된다.", 0],
    ["계약 당사자는 상호 존중을 바탕으로 계약을 이행하며, 필요 시 협의를 통해 문제를 해결한다.", 0],
    ["계약 종료 후에도 당사자는 일정 기간 동안 기밀 유지 의무를 준수해야 한다.", 0],
    ["본 계약에서 명시되지 않은 사항은 관련 법령 및 일반적인 상거래 관행을 따른다.", 0],
]


from torch.utils.data import DataLoader, TensorDataset
import torch
train_encodings = tokenizer(
    dataset["train"]["text"], padding="max_length", truncation=True, max_length=256, return_tensors="pt"
)
test_encodings = tokenizer(
    dataset["test"]["text"], padding="max_length", truncation=True, max_length=256, return_tensors="pt"
)
# ✅ 레이블을 Tensor로 변환
train_labels = torch.tensor(dataset["train"]["label"], dtype=torch.float32).unsqueeze(1)  # [batch, 1] 형태
test_labels = torch.tensor(dataset["test"]["label"], dtype=torch.float32).unsqueeze(1)

# ✅ TensorDataset 생성
train_tensor_dataset = TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"], train_labels)
test_tensor_dataset = TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"], test_labels)

# ✅ DataLoader 생성
batch_size = 16  # 배치 크기 설정 (필요하면 변경 가능)
train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_tensor_dataset, batch_size=batch_size, shuffle=False)

"""
import os, sys
sys.path.append(os.path.abspath("./AI"))
import threshold_settings as ts
threshold= ts.find_threshold_2(model, train_loader=train_loader, val_loader=val_loader, use_train=False, device=device)
최적 임계값: 0.6134
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
y_pred = []
y_true = []
threshold = 0.6134

# 테스트 실행 및 결과 출력
print("\n🔹 **테스트 데이터 예측 결과** 🔹\n")
for text, label in test_data:
    probability = predict(text)
    print(f"문장: {text}")
    print(f"예측 독소 조항 확률: {probability*100:.2f}%")
    print("-" * 80)
    y_pred.append(1 if probability >= threshold else 0)
    y_true.append(label)

# ✅ 성능 지표 계산
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)

# ✅ 성능 지표 출력
print("📊 테스트 데이터 성능 평가 결과 📊")
print(f"📌 최적 임계값: {threshold:.4f}")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1-score: {f1:.4f}")
print(f"✅ ROC-AUC: {roc_auc:.4f}")