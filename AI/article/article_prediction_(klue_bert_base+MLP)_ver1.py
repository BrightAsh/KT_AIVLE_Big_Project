import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os



name = 'article_prediction_(klue_bert_base+MLP)_ver1_2차'

# ✅ 저장할 디렉토리 설정 (폴더 없으면 생성)
save_path = f"./Data_Analysis/Model/{name}"
os.makedirs(save_path, exist_ok=True)
model_file = os.path.join(save_path, "klue_bert_mlp.pth")


# ✅ KLUE/BERT 토크나이저 및 모델 로드
model_name = "klue/bert-base"
tokenizer = BertTokenizer.from_pretrained(model_name)

# ✅ 데이터 로드 (sentence, label, article 포함)
df = pd.read_csv('./Data_Analysis/Data/unfair_sentence_merged.csv')  # sentence, label, article 컬럼

# ✅ 불공정 조항만 필터링 (label == 1인 문장만 사용)
df_unfair = df[df["label"] == 1].reset_index(drop=True)

# ✅ Article을 숫자로 매핑 (예: 제26조 → 0, 제29조 → 1 ...)
article_to_idx = {article: idx for idx, article in enumerate(df_unfair["article"].unique())}
idx_to_article = {idx: article for article, idx in article_to_idx.items()}

# ✅ Train/Test 데이터 분할 (8:2 비율)
X_train, X_val, y_train, y_val = train_test_split(
    df_unfair["sentence"].tolist(), df_unfair["article"].map(article_to_idx).tolist(),
    test_size=0.2, random_state=42, stratify=df_unfair["article"]
)

# ✅ 토큰화 및 텐서 변환
def tokenize_data(sentences, tokenizer, max_length=256):
    encoding = tokenizer(
        sentences, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    return encoding["input_ids"], encoding["attention_mask"]

# ✅ 훈련 및 검증 데이터 토큰화
X_train_ids, X_train_mask = tokenize_data(X_train, tokenizer)
X_val_ids, X_val_mask = tokenize_data(X_val, tokenizer)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

from torch.utils.data import DataLoader, TensorDataset
batch_size = 16
train_dataset = TensorDataset(X_train_ids, X_train_mask, y_train_tensor)
val_dataset = TensorDataset(X_val_ids, X_val_mask, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# ✅ 조항 개수 (출력 클래스 수)
num_articles = len(article_to_idx)

# ✅ KLUE/BERT + MLP (Multi-Class Classification)
class BertArticleClassifier(nn.Module):
    def __init__(self, bert_model_name="klue/bert-base", hidden_size=256, num_classes=num_articles):
        super(BertArticleClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 조항 개수만큼 출력
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 벡터 사용
        x = self.fc1(cls_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)  # 확률 분포 출력

# ✅ 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
article_model = BertArticleClassifier().to(device)

criterion = nn.CrossEntropyLoss()  # Multi-Class Classification
optimizer = optim.Adam(article_model.parameters(), lr=0.00002)

# ✅ ReduceLROnPlateau 추가 (patience=2, factor=0.5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)


# ✅ 모델 학습 함수 (Early Stopping 포함)
import matplotlib.pyplot as plt

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

# ✅ 모델 학습 함수 수정
def train_article_model(model, train_loader, val_loader, epochs=10, patience=3):
    best_loss = float('inf')
    patience_counter = 0
    train_loss_list = []
    val_loss_list = []

    # ✅ ReduceLROnPlateau 스케줄러 추가
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
    loss_plot_path = os.path.join(save_path, "loss_curve_article.png")
    plot_loss_curve(train_loss_list, val_loss_list, loss_plot_path)


# ✅ 모델 학습 실행
train_article_model(article_model, train_loader, val_loader, epochs=1000, patience=10)

# ✅ 모델 저장 (state_dict만 저장)
torch.save(article_model.state_dict(), model_file)
print(f"✅ 모델 저장 완료: {model_file}")

# ✅ 법률 조항 예측 함수
def predict_article(a_model,sentence):
    """문장 입력 시, 가장 관련 있는 법률 조항(Article)을 예측"""
    a_model.eval()
    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)

    with torch.no_grad():
        output = a_model(inputs["input_ids"], inputs["attention_mask"])
        predicted_idx = torch.argmax(output).item()  # 가장 확률 높은 클래스 선택
        predicted_article = idx_to_article[predicted_idx]  # 조항명으로 변환

    return {
        "sentence": sentence,
        "predicted_article": predicted_article
    }




# ✅ 저장된 모델 로드 함수
def load_article_model(model_file):
    model = BertArticleClassifier().to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    print(f"✅ 저장된 모델 로드 완료: {model_file}")
    return model

# ✅ 모델 로드 실행
loaded_article_model = load_article_model(model_file)


# ✅ 테스트 실행
test_sentences = [
    ('갑은 을과의 분쟁이 발생하더라도 협의회의 조정 절차를 무시할 수 있다.', 26),  # 제26조
    ('을이 신고를 취하한 경우라도, 공정거래위원회는 신고 사실을 계속 유지해야 한다.', 29),  # 제29조
    ('공정거래위원회는 서면실태조사 결과를 공표하지 않아도 된다.', 30),  # 제30조
    ('갑은 공정거래위원회의 조정 절차가 진행 중이더라도 이를 무시하고 독단적으로 결정을 내릴 수 있다.', 26),  # 제26조
    ('을이 신고를 했더라도, 갑은 공정거래위원회의 조사에 협조하지 않을 권리가 있다.', 29),  # 제29조
    ('협의회는 조정 신청이 접수되었더라도 분쟁당사자에게 통보하지 않아도 된다.', 25),  # 제25조
    ('갑은 협의회의 조정 절차를 따르지 않고 자체적으로 해결 방안을 강요할 수 있다.', 28),  # 제28조
    ('공정거래위원회는 갑이 위반 혐의를 받더라도 직권 조사를 하지 않아도 된다.', 29),  # 제29조
    ('갑은 을에게 서면실태조사와 관련된 자료 제출을 거부하도록 강요할 수 있다.', 30),  # 제30조
    ('조정조서는 법적 효력이 없으므로 갑은 이를 따를 필요가 없다.', 27),  # 제27조
]
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

y_pred = []
y_true = []

# ✅ 수정된 루프 (문장과 레이블을 분리하여 사용)
for sentence, label in test_sentences:
    result = predict_article(loaded_article_model, sentence)  # 한 문장씩 예측 수행
    print(f"{result['predicted_article']}/{label}")

    y_pred.append(result['predicted_article'])
    y_true.append(label)

# ✅ 성능 지표 계산 (다중 클래스 설정 추가)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro", zero_division=1)  # ✅ 수정됨
recall = recall_score(y_true, y_pred, average="macro", zero_division=1)  # ✅ 수정됨
f1 = f1_score(y_true, y_pred, average="macro", zero_division=1)  # ✅ 수정됨

# ✅ ROC-AUC 예외 처리 (다중 클래스 지원)
try:
    roc_auc = roc_auc_score(y_true, y_pred, multi_class="ovr")
except ValueError:
    roc_auc = float('nan')

# ✅ 성능 지표 출력
print("\n📊 테스트 데이터 성능 평가 결과 📊")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1-score: {f1:.4f}")
print(f"✅ ROC-AUC: {roc_auc:.4f}")
