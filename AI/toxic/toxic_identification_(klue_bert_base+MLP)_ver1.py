import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.model_selection import train_test_split
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
name = 'toxic_(klue_bert_base_MLP)_ver1_1차'
model_name = "klue/bert-base"
tokenizer = BertTokenizer.from_pretrained(model_name)


save_path = f"./Data_Analysis/Model/{name}/"
os.makedirs(save_path, exist_ok=True)
model_file = os.path.join(save_path, "klue_bert_mlp.pth")


class BertMLPClassifier(nn.Module):
    def __init__(self, bert_model_name="klue/bert-base", hidden_size=256):
        super(BertMLPClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, 1)  # 불공정(1) 확률을 출력
        self.sigmoid = nn.Sigmoid()  # 확률값으로 변환

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 벡터 사용
        x = self.fc1(cls_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)  # 0~1 확률값 반환


data = pd.read_csv('./Data_Analysis/Data/toxic_sentence_merged.csv')

print(f'data_shape: {data.shape}')

X_train, X_val, y_train, y_val = train_test_split(data["sentence"].tolist(), data["label"].tolist(), test_size=0.2, random_state=42, stratify=data["label"],shuffle=True)


def tokenize_data(sentences, tokenizer, max_length=256):
    encoding = tokenizer(
        sentences, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    return encoding["input_ids"], encoding["attention_mask"]


X_train_ids, X_train_mask = tokenize_data(X_train, tokenizer)
X_val_ids, X_val_mask = tokenize_data(X_val, tokenizer)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # [batch, 1] 형태
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)


# ✅ 모델 초기화

model = BertMLPClassifier().to(device)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss 사용
optimizer = optim.Adam(model.parameters(), lr=0.00002)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

# ✅ 모델 학습 함수 (Early Stopping 적용, 저장 X)
from torch.utils.data import DataLoader, TensorDataset

# ✅ 배치 크기 설정
batch_size = 16

# ✅ 데이터셋 & 데이터로더 설정
train_dataset = TensorDataset(X_train_ids, X_train_mask, y_train_tensor)
val_dataset = TensorDataset(X_val_ids, X_val_mask, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#######################################################################################################################
# train
#######################################################################################################################
import matplotlib.pyplot as plt
# ✅ 모델 학습 함수 수정
# ✅ 손실 그래프 저장 함수
def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)  # 이미지로 저장
    print(f"✅ Loss 그래프 저장 완료: {save_path}")
    plt.close()

# ✅ 모델 학습 함수 수정 (손실 그래프 추가)
def train_model(model, train_loader, val_loader, epochs=10, patience=3):
    best_loss = float('inf')
    patience_counter = 0
    train_loss_list = []
    val_loss_list = []

    # ✅ ReduceLROnPlateau 스케줄러 추가 (patience=2, factor=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # ✅ 배치 학습 적용
        for X_batch, mask_batch, y_batch in train_loader:
            X_batch, mask_batch, y_batch = X_batch.to(device), mask_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch, mask_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ✅ 검증 단계
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, mask_batch, y_batch in val_loader:
                X_batch, mask_batch, y_batch = X_batch.to(device), mask_batch.to(device), y_batch.to(device)
                val_outputs = model(X_batch, mask_batch)
                val_loss += criterion(val_outputs, y_batch).item()

        val_loss /= len(val_loader)

        # ✅ 손실 저장
        train_loss_list.append(total_loss)
        val_loss_list.append(val_loss)

        # ✅ ReduceLROnPlateau 적용
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # ✅ Early Stopping 체크
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # ✅ 손실 그래프 저장
    loss_plot_path = os.path.join(save_path, "loss_curve.png")
    plot_loss_curve(train_loss_list, val_loss_list, loss_plot_path)

# ✅ 모델 학습 실행 (손실 그래프 추가됨)
train_model(model, train_loader, val_loader, epochs=1000, patience=10)

torch.save(model.state_dict(), model_file)
print(f"✅ 모델 저장 완료: {model_file}")
#######################################################################################################################
#######################################################################################################################

def predict_toxic_clause(c_model, sentence, threshold=0.5):
    c_model.eval()
    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        output = c_model(inputs["input_ids"], inputs["attention_mask"])
        unfair_prob = output.item()
    return {
        "sentence": sentence,
        "toxic_probability": round(unfair_prob * 100, 2),  # 1(불공정) 확률
        "predicted_label": "독소" if unfair_prob >= threshold else "비독소"
    }

test_data = [
    ("을은 갑의 요청이 있을 경우, 정해진 계약 기간과 관계없이 추가적인 납품을 진행해야 한다.", 1),
    ("갑은 을의 재고 상황과 관계없이 주문량을 자유롭게 조정할 수 있으며, 을은 이에 무조건 응해야 한다.", 1),
    ("을은 갑의 판매 전략에 따라 원가 이하의 가격으로 납품해야 하며, 이에 대한 손실 보전을 요구할 수 없다.", 1),
    ("본 계약 체결 이후에도 갑은 을의 유통망을 직접 통제할 수 있으며, 을은 이를 거부할 수 없다.", 1),
    ("을은 갑의 경영 전략에 따라 가격 및 판매 정책을 조정해야 하며, 이에 대한 협의 권한이 없다.", 1),
    ("갑은 을의 납품 기한을 사전 협의 없이 조정할 수 있으며, 을은 이에 즉시 응해야 한다.", 1),
    ("을은 갑의 판매 촉진을 위해 추가적인 제품을 무상으로 제공해야 하며, 이에 대한 대가를 요구할 수 없다.", 1),
    ("본 계약의 종료 여부는 갑이 단독으로 결정하며, 을은 이에 대해 어떠한 권리도 주장할 수 없다.", 1),
    ("갑은 을의 생산 과정에 개입할 권리를 가지며, 을은 이에 대해 거부할 수 없다.", 1),
    ("을은 계약이 종료된 후에도 일정 기간 동안 갑이 요청하는 조건을 유지하여 제품을 공급해야 한다.", 1),
    ("계약 당사자는 계약의 이행을 위해 상호 협력하며, 문제 발생 시 협의를 통해 해결해야 한다.", 0),
    ("을은 계약된 일정에 따라 제품을 납품하며, 일정 변경이 필요한 경우 사전에 협의한다.", 0),
    ("본 계약의 조항은 양측의 동의 없이 일방적으로 변경될 수 없다.", 0),
    ("계약 해지 시, 당사자는 합의된 절차에 따라 서면으로 통보해야 한다.", 0),
    ("갑은 을의 정당한 사유 없이 계약 조건을 일방적으로 변경할 수 없다.", 0),
    ("을은 계약 이행 중 발생하는 문제를 갑에게 즉시 보고하고 협의해야 한다.", 0),
    ("본 계약은 계약서에 명시된 기한 동안 적용되며, 연장은 양측 협의를 통해 진행된다.", 0),
    ("계약 당사자는 상호 존중을 바탕으로 계약을 이행하며, 필요 시 협의를 통해 문제를 해결한다.", 0),
    ("계약 종료 후에도 당사자는 일정 기간 동안 기밀 유지 의무를 준수해야 한다.", 0),
    ("본 계약에서 명시되지 않은 사항은 관련 법령 및 일반적인 상거래 관행을 따른다.", 0),
]

def load_trained_model(model_file):
    # ✅ 모델 객체를 새로 생성한 후 state_dict만 로드해야 함
    model = BertMLPClassifier().to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    print(f"✅ 저장된 모델 로드 완료: {model_file}")
    return model
loaded_model = load_trained_model(model_file)

"""
import os, sys
sys.path.append(os.path.abspath("./AI"))
import threshold_settings as ts
threshold= ts.find_threshold(loaded_model, train_loader=train_loader, val_loader=val_loader, use_train=False, device=device)
최적 임계값: 0.5011
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
y_pred = []
y_true = []
threshold = 0.5011

for sentence, label in test_data:
    result = predict_toxic_clause(loaded_model,sentence,threshold)
    print(f"📝 계약 조항: {result['sentence']}")
    print(f"🔍 판별 결과: {result['predicted_label']} (독소 확률: {result['toxic_probability']}%)")
    print(f"✅ 정답: {'독소' if label == 1 else '합법'}")
    print("-" * 50)
    y_pred.append(1 if result['toxic_probability'] >= threshold else 0)
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