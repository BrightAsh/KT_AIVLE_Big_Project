from sklearn.cluster import KMeans
import numpy as np
import torch
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score
from transformers import BertTokenizer, BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_threshold(model, train_loader=None, val_loader=None, use_train=False, device="cuda"):
    model.eval()
    y_probs, y_true = [], []
    def collect_predictions(loader):
        with torch.no_grad():
            for X_batch, mask_batch, y_batch in loader:
                X_batch, mask_batch, y_batch = X_batch.to(device), mask_batch.to(device), y_batch.to(device)
                outputs = model(X_batch, mask_batch)
                y_probs.extend(outputs.cpu().numpy().flatten())  # 확률값 저장
                y_true.extend(y_batch.cpu().numpy().flatten())  # 실제 레이블 저장

    # ✅ Train 데이터 포함 여부 선택
    if use_train and train_loader:
        collect_predictions(train_loader)
    if val_loader:
        collect_predictions(val_loader)

    y_probs = np.array(y_probs)
    y_true = np.array(y_true)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(y_probs.reshape(-1, 1))
    cluster_centers = np.sort(kmeans.cluster_centers_.flatten())  # 군집 중심 정렬
    optimal_threshold = np.mean(cluster_centers)  # 두 군집 중심의 중간값

    print(f"📌 최적 임계값: {optimal_threshold:.4f}")
    return optimal_threshold

##########################################################################################################################################################################################
##########################################################################################################################################################################################
##########################################################################################################################################################################################
##########################################################################################################################################################################################

def find_threshold_2(model, train_loader=None, val_loader=None, use_train=False, device="cuda"):
    model.eval()
    y_probs, y_true = [], []
    def collect_predictions(loader):
        with torch.no_grad():
            for X_batch, mask_batch, y_batch in loader:
                X_batch, mask_batch, y_batch = X_batch.to(device), mask_batch.to(device), y_batch.to(device)
                outputs = model(X_batch, mask_batch)
                y_probs.extend(torch.sigmoid(outputs.logits).cpu().numpy().flatten())  # ✅ 수정된 코드
                y_true.extend(y_batch.cpu().numpy().flatten())  # 실제 레이블 저장

    # ✅ Train 데이터 포함 여부 선택
    if use_train and train_loader:
        collect_predictions(train_loader)
    if val_loader:
        collect_predictions(val_loader)

    y_probs = np.array(y_probs)
    y_true = np.array(y_true)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(y_probs.reshape(-1, 1))
    cluster_centers = np.sort(kmeans.cluster_centers_.flatten())  # 군집 중심 정렬
    optimal_threshold = np.mean(cluster_centers)  # 두 군집 중심의 중간값

    print(f"📌 최적 임계값: {optimal_threshold:.4f}")
    return optimal_threshold

