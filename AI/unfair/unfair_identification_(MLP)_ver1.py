from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import random
import pandas as pd

name = '2025_01_08_unfair_ver1_1차'
# 1️⃣ 법률 데이터 로드 (JSON 구조 활용)
with open("./Data_Analysis/Data/law_data.json", "r", encoding="utf-8") as f:
    law_data = json.load(f)

# 2️⃣ BERT 모델 및 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
bert_model = BertModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def embed_texts(texts):
    """텍스트 리스트를 BERT 임베딩 벡터로 변환"""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # [CLS] 토큰의 벡터 사용


# 3️⃣ 법률 조항 및 항목 벡터화
law_embeddings = []
law_info = []

for law in law_data:
    article_text = law["article_content"].strip()
    if article_text:
        law_embeddings.append(embed_texts([article_text])[0])
        law_info.append({
            "law_reference": f"{law['article_number']} {law['article_title']}",
            "content": article_text
        })

    for clause in law["clauses"]:
        clause_text = clause["content"].strip()
        if clause_text:
            law_embeddings.append(embed_texts([clause_text])[0])
            law_info.append({
                "law_reference": f"{law['article_number']} {clause['clause_number']}",
                "content": clause_text
            })

law_embeddings = np.array(law_embeddings)

# 4️⃣ 계약서 문장 데이터 로드 (CSV -> DataFrame)
contract_data = pd.read_csv('./Data_Analysis/Data/unfair_sentence(MJ).csv')

# 5️⃣ 계약서 문장 벡터화 (sentence 컬럼 사용)
X_train = embed_texts(contract_data["sentence"].tolist())
y_train = contract_data["label"].tolist()

from sklearn.model_selection import train_test_split

# 1️⃣ Train/Validation 데이터 분할 (8:2 비율)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
, stratify=y_train)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# 6️⃣ MLP 분류 모델 정의
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        return self.softmax(x)


input_size = X_train.shape[1]
hidden_size = 512
output_size = 2  # 0: 합법, 1: 불공정

classifier = MLPClassifier(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)


# 7️⃣ MLP 모델 학습 (Early Stopping 포함)
# 3️⃣ MLP 모델 학습 (Validation 데이터 포함)
def train_model(X_train, y_train, X_val, y_val, model, criterion, optimizer, epochs=1000, patience=20):
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # 3-1. 모델 훈련 (Train)
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # 3-2. 검증 단계 (Validation)
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        # 3-3. Early Stopping 체크
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            patience_counter = 0  # 개선되었으므로 patience 초기화
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"✅ Early stopping at epoch {epoch + 1}, Validation Loss: {val_loss.item():.4f}")
            break

        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

# 4️⃣ 모델 학습 실행 (Train & Validation 적용)
train_model(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, classifier, criterion, optimizer, epochs=1000, patience=20)


def find_most_similar_law(contract_text):
    """계약서 문장과 가장 유사한 법률 조항 찾기"""
    contract_embedding = embed_texts([contract_text])[0]
    similarities = cosine_similarity([contract_embedding], law_embeddings)[0]
    best_idx = np.argmax(similarities)
    return law_info[best_idx], similarities[best_idx]


def predict_contract_legality(contract_text):
    """계약서 문장을 입력하면 불공정 여부와 위반 가능성이 높은 법률 조항 반환"""
    similar_law, similarity_score = find_most_similar_law(contract_text)
    contract_embedding = torch.tensor(embed_texts([contract_text])[0], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = classifier(contract_embedding)
    unfair_prob = output[0][1].item()

    return {
        "contract_text": contract_text,
        "predicted_label": "불공정" if unfair_prob > 0.5 else "합법",
        "unfair_probability": round(unfair_prob * 100, 2),
        "violated_law": similar_law["law_reference"],
        "similar_law_content": similar_law["content"],
        "similarity_score": round(similarity_score * 100, 2)
    }


# 7️⃣ 테스트 실행 (test_data 사용)
test_data =[
    ['갑은 을과의 분쟁이 발생하더라도 협의회의 조정 절차를 무시할 수 있다.',1],	#제26조
    ['을이 신고를 취하한 경우라도, 공정거래위원회는 신고 사실을 계속 유지해야 한다.',1],	#제29조
    ['공정거래위원회는 서면실태조사 결과를 공표하지 않아도 된다.',1],	#제30조
    ['갑은 공정거래위원회의 조정 절차가 진행 중이더라도 이를 무시하고 독단적으로 결정을 내릴 수 있다.',1],	#제26조
    ['을이 신고를 했더라도, 갑은 공정거래위원회의 조사에 협조하지 않을 권리가 있다.',1],	#제29조
    ['협의회는 조정 신청이 접수되었더라도 분쟁당사자에게 통보하지 않아도 된다.',1],	#제25조
    ['갑은 협의회의 조정 절차를 따르지 않고 자체적으로 해결 방안을 강요할 수 있다.',1],	#제28조
    ['공정거래위원회는 갑이 위반 혐의를 받더라도 직권 조사를 하지 않아도 된다.',1],	#제29조
    ['갑은 을에게 서면실태조사와 관련된 자료 제출을 거부하도록 강요할 수 있다.',1],	#제30조
    ['조정조서는 법적 효력이 없으므로 갑은 이를 따를 필요가 없다.',1],	#제27조
    ['공정거래위원회는 직권으로 대규모유통업자의 위반 행위를 조사할 수 있다.',0],
    ['협의회는 조정 신청을 받으면 즉시 조정 절차를 개시해야 한다.',0],
    ['갑과 을은 협의회의 조정 절차를 성실히 따라야 한다.',0],
    ['누구든지 이 법을 위반하는 사실을 발견하면 공정거래위원회에 신고할 수 있다.',0],
    ['협의회는 서면실태조사 결과를 공정하게 공개해야 한다.',0],
    ['조정조서는 재판상 화해와 동일한 효력을 가지므로 반드시 이행되어야 한다.',0],
    ['서면실태조사는 공정한 거래질서 확립을 위해 반드시 시행되어야 한다.',0],
    ['협의회의 운영 절차는 공정성을 보장할 수 있도록 대통령령에 따라야 한다.',0],
    ['공정거래위원회는 법에 따라 갑의 위반 혐의를 조사할 수 있다.',0],
    ['협의회의 조정 절차가 종료되면 시효는 새롭게 진행된다.',0],
]

toxic_test_data = [
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

results = []
for sentence, label in toxic_test_data:
    result = predict_contract_legality(sentence)
    results.append(result)
    print(f"📝 계약 조항: {result['contract_text']}")
    print(f"🔍 판별 결과: {result['predicted_label']} (불공정 확률: {result['unfair_probability']}%)")
    print(f"⚖ 위반 가능성이 있는 법률: {result['violated_law']}")
    print(f"📜 관련 법 조항: {result['similar_law_content']}")
    print(f"🔗 유사도 점수: {result['similarity_score']}%")
    print(f"✅ 정답: {'불공정' if label == 1 else '합법'}")
    print("-" * 50)


torch.save(classifier.state_dict(), f"./Data_Analysis/Model/{name}/mlp_classifier.pth")
np.save("./Data_Analysis/Data/law_embeddings.npy", law_embeddings)

