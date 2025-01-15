"""
1. 학습 데이터 불균형 문제 해결
불공정 조항과 독소 조항은 대부분 불균형 데이터일 가능성이 높음 (일반 조항이 더 많을 수도 있음)
불균형 데이터 문제를 해결하면 모델이 한쪽으로 치우치지 않고 일반화 성능이 좋아짐
➡ 해결 방법:
✔ Focal Loss 적용
✔ Weighted Random Sampler를 사용하여 학습 데이터의 불균형을 완화

✔ Focal Loss 적용 (CrossEntropyLoss 대신 사용)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCELoss()

    def forward(self, inputs, targets):
        BCE_loss = self.bce_loss(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

➡ criterion = FocalLoss() 로 변경하여 학습 진행
✔ Weighted Random Sampler 적용 (데이터 비율 맞추기)
from torch.utils.data.sampler import WeightedRandomSampler

# 클래스별 샘플 개수 계산
class_sample_counts = [sum(y_train), len(y_train) - sum(y_train)]
weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
sample_weights = weights[y_train_tensor.squeeze().long()]

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
➡ 데이터 균형을 맞추면서 학습 진행 가능

2. 혼동 행렬(Confusion Matrix) 분석
학습 후 모델의 예측 결과를 분석하려면 혼동 행렬(Confusion Matrix) 을 확인하는 것이 중요
단순히 loss 값만 보는 것이 아니라 오분류 패턴을 분석하여 모델을 개선 가능
➡ 해결 방법:
✔ sklearn.metrics.confusion_matrix를 활용하여 예측 결과 비교
✔ classification_report 로 정확도, 정밀도, 재현율 확인
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, data_loader):
    model.eval()
    y_preds, y_trues = [], []

    with torch.no_grad():
        for X_batch, mask_batch, y_batch in data_loader:
            X_batch, mask_batch, y_batch = X_batch.to(device), mask_batch.to(device), y_batch.to(device)
            outputs = model(X_batch, mask_batch)
            preds = (outputs >= 0.5).float()  # 0.5 이상이면 1(독소 조항), 미만이면 0(일반 조항)
            y_preds.extend(preds.cpu().numpy())
            y_trues.extend(y_batch.cpu().numpy())

    # ✅ 혼동 행렬 출력
    cm = confusion_matrix(y_trues, y_preds)
    print("\n🔹 Confusion Matrix:")
    print(cm)

    # ✅ 정확도, 정밀도, 재현율 출력
    report = classification_report(y_trues, y_preds, target_names=["정상 조항", "독소 조항"])
    print("\n🔹 Classification Report:")
    print(report)

# ✅ 학습 완료 후 평가 실행
evaluate_model(loaded_model, val_loader)


"""