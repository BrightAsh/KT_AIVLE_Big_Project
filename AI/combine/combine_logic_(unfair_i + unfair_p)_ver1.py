import numpy as np
import json
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn

#####################################################################################################################################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 1️⃣ KLUE BERT 모델 및 토크나이저 로드
MODEL_NAME = "klue/bert-base"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertModel.from_pretrained(MODEL_NAME).to(device)

import  pandas as pd
df = pd.read_csv('./Data_Analysis/Data/unfair_sentence(MJ)+article.csv')  # sentence, label, article 컬럼
df_unfair = df[df["label"] == 1].reset_index(drop=True)
article_to_idx = {article: idx for idx, article in enumerate(df_unfair["article"].unique())}
idx_to_article = {idx: article for article, idx in article_to_idx.items()}
num_articles = len(article_to_idx)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 1️⃣ KLUE BERT 기반 불공정 조항 판별 모델 정의
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
# ✅ 2️⃣ KLUE BERT 기반 법률 조항 예측 모델 정의
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


# ✅ 3️⃣ 학습된 모델 로드 함수 (가중치 적용)
def load_trained_model_statice(model_class, model_file):
    """저장된 모델 가중치를 올바른 모델 구조에 적용하여 로드"""
    model = model_class().to(device)  # 모델 초기화
    state_dict = torch.load(model_file, map_location=device)  # 가중치 로드
    if isinstance(state_dict, dict):  # 올바른 state_dict인지 확인
        model.load_state_dict(state_dict, strict=False)  # strict=False로 예외 방지
        model.eval()  # 평가 모드 설정
        print(f"✅ 저장된 모델 로드 완료: {model_file}")
        return model
    else:
        raise TypeError(f"❌ 모델 가중치 로드 실패: {model_file} (잘못된 데이터 타입 {type(state_dict)})")

## 임시 함수
def load_trained_model_directly(model_file):
    """모델 전체를 저장한 경우 직접 로드하여 반환"""
    model = torch.load(model_file, map_location=device)  # 모델 전체 로드
    model.to(device)  # GPU/CPU 설정
    model.eval()  # 평가 모드 설정
    print(f"✅ 저장된 모델 전체 로드 완료: {model_file}")
    return model

# ✅ 5️⃣ 법률 조항 예측 모델 로드
article_model = load_trained_model_statice(BertArticleClassifier, "./Data_Analysis/Model/2025_01_09_article_prediction_ver2_1차/klue_bert_mlp.pth")
# ✅ 3️⃣ 불공정 조항 판별 모델 로드
unfair_model = load_trained_model_directly("./Data_Analysis/Model/2025_01_09_unfair_identification_ver2_2차/klue_bert_mlp.pth")

# ✅ 5️⃣ 법률 데이터 로드 (JSON)
with open("./Data_Analysis/Data/law_data.json", "r", encoding="utf-8") as f:
    law_data = json.load(f)

# ✅ 6️⃣ 법률 조항 벡터화 (항/호 포함)
law_embeddings = []
law_info = []

for law in law_data:
    article_text = law["article_content"].strip()
    if article_text:
        law_embeddings.append(bert_model(
            **tokenizer(article_text, return_tensors="pt", padding=True, truncation=True).to(
                device)).pooler_output.cpu().detach().numpy()[0])
        law_info.append({
            "law_reference": f"{law['article_number']}",
            "content": article_text
        })

    for clause in law["clauses"]:
        clause_text = clause["content"].strip()
        if clause_text:
            law_embeddings.append(bert_model(
                **tokenizer(clause_text, return_tensors="pt", padding=True, truncation=True).to(
                    device)).pooler_output.cpu().detach().numpy()[0])
            law_info.append({
                "law_reference": f"{law['article_number']} {clause['clause_number']}",
                "content": clause_text
            })

law_embeddings = np.array(law_embeddings)

#####################################################################################################################################################################################################################

# ✅ 7️⃣ 불공정 조항 판별 함수
def predict_unfair_clause(model, sentence, threshold=0.5):
    model.eval()
    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(inputs["input_ids"], inputs["attention_mask"])
        unfair_prob = output.item()

    return {
        "sentence": sentence,
        "unfair_probability": round(unfair_prob * 100, 2),  # 1(불공정) 확률
        "predicted_label": "불공정" if unfair_prob >= threshold else "합법"
    }


def predict_article(model,sentence):
    model.eval()
    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model(inputs["input_ids"], inputs["attention_mask"])
        predicted_idx = torch.argmax(output).item()  # 가장 확률 높은 클래스 선택
        predicted_article = idx_to_article[predicted_idx]  # 조항명으로 변환

    return predicted_article


"""
# ✅ 9️⃣ 법 조항 내 가장 유사한 항/호 찾기
def find_most_similar_law_within_article(sentence, predicted_article):
    #예측된 조항 내에서 가장 유사한 항/호 찾기
    contract_embedding = bert_model(**tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(
        device)).pooler_output.cpu().detach().numpy()[0]

    # 해당 조항 내 항/호 필터링
    filtered_law_embeddings = []
    filtered_law_info = []

    for info, embedding in zip(law_info, law_embeddings):
        if info["law_reference"].split()[1] == str(predicted_article):  # 예측된 조항 포함 여부 확인
            filtered_law_embeddings.append(embedding)
            filtered_law_info.append(info)

    if not filtered_law_embeddings:  # 예측 조항 내 항/호가 없을 경우
        return predicted_article, "조 내 상세 항/호 없음"

    # 유사도 계산 및 최적 항/호 선택
    similarities = cosine_similarity([contract_embedding], filtered_law_embeddings)[0]
    best_idx = np.argmax(similarities)

    return filtered_law_info[best_idx]["law_reference"], filtered_law_info[best_idx]["content"]
"""


def find_most_similar_law_within_article(sentence, predicted_article):
    """예측된 조항 내에서 가장 유사한 항/호 찾기"""
    contract_embedding = bert_model(**tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(
        device)).pooler_output.cpu().detach().numpy()[0]

    predicted_article = str(predicted_article)  # 숫자형 조항을 문자열로 변환

    # 해당 조항 탐색
    matching_article = None
    for article in law_data:
        if article["article_number"] == f"Article {predicted_article}":
            matching_article = article
            break

    if not matching_article:
        return {
            "Article number": None,
            "Article title": None,
            "Paragraph number": None,
            "Subparagraph number": None,
            "Article detail": None,
            "Paragraph detail": None,
            "Subparagraph detail": None
        }

    # 조항 제목과 내용 추출
    article_title = matching_article.get("article_title", None)
    article_detail = matching_article.get("article_content", None) if matching_article.get("article_content") else None

    # 항(Paragraph)과 호(Subparagraph) 필터링
    filtered_law_embeddings = []
    filtered_law_info = []

    for clause in matching_article["clauses"]:
        clause_text = clause["content"].strip()
        clause_number = clause["clause_number"]

        if clause_text:
            clause_embedding = bert_model(
                **tokenizer(clause_text, return_tensors="pt", padding=True, truncation=True).to(device)
            ).pooler_output.cpu().detach().numpy()[0]

            filtered_law_embeddings.append(clause_embedding)
            filtered_law_info.append({
                "clause_number": clause_number,
                "clause_content": clause_text,
                "sub_clauses": clause.get("sub_clauses", [])
            })

    if not filtered_law_embeddings:
        return {
            "Article number": f"Article {predicted_article}",
            "Article title": article_title,
            "Paragraph number": None,
            "Subparagraph number": None,
            "Article detail": article_detail,
            "Paragraph detail": None,
            "Subparagraph detail": None
        }

    # 유사도 계산 및 최적 항/호 선택
    similarities = cosine_similarity([contract_embedding], filtered_law_embeddings)[0]
    best_idx = np.argmax(similarities)
    best_match = filtered_law_info[best_idx]

    paragraph_number = best_match["clause_number"]
    paragraph_detail = best_match["clause_content"]

    # 가장 적합한 호(Subparagraph) 찾기
    best_subparagraph = None
    if best_match["sub_clauses"]:
        for sub in best_match["sub_clauses"]:
            if isinstance(sub, str):  # 서브클로즈가 단순 텍스트일 경우
                best_subparagraph = sub
                break

    return {
        "Article number": f"Article {predicted_article}",
        "Article title": article_title,
        "Paragraph number": f"Paragraph {paragraph_number}" if paragraph_number else None,
        "Subparagraph number": None if not best_subparagraph else "Subparagraph",
        "Article detail": article_detail,
        "Paragraph detail": paragraph_detail,
        "Subparagraph detail": best_subparagraph
    }


# ✅ 1️⃣0️⃣ 테스트 실행
test_sentences = [
    ('갑은 을과의 분쟁이 발생하더라도 협의회의 조정 절차를 무시할 수 있다.', 1, 26),  # 제26조
    ('을이 신고를 취하한 경우라도, 공정거래위원회는 신고 사실을 계속 유지해야 한다.', 1, 29),  # 제29조
    ('공정거래위원회는 서면실태조사 결과를 공표하지 않아도 된다.', 1, 30),  # 제30조
    ('갑은 공정거래위원회의 조정 절차가 진행 중이더라도 이를 무시하고 독단적으로 결정을 내릴 수 있다.', 1, 26),  # 제26조
    ('을이 신고를 했더라도, 갑은 공정거래위원회의 조사에 협조하지 않을 권리가 있다.', 1, 29),  # 제29조
    ('협의회는 조정 신청이 접수되었더라도 분쟁당사자에게 통보하지 않아도 된다.', 1, 25),  # 제25조
    ('갑은 협의회의 조정 절차를 따르지 않고 자체적으로 해결 방안을 강요할 수 있다.', 1, 28),  # 제28조
    ('공정거래위원회는 갑이 위반 혐의를 받더라도 직권 조사를 하지 않아도 된다.', 1, 29),  # 제29조
    ('갑은 을에게 서면실태조사와 관련된 자료 제출을 거부하도록 강요할 수 있다.', 1, 30),  # 제30조
    ('조정조서는 법적 효력이 없으므로 갑은 이를 따를 필요가 없다.', 1, 27),  # 제27조
    ('공정거래위원회는 직권으로 대규모유통업자의 위반 행위를 조사할 수 있다.', 0, 0),
    ('협의회는 조정 신청을 받으면 즉시 조정 절차를 개시해야 한다.', 0, 0),
    ('갑과 을은 협의회의 조정 절차를 성실히 따라야 한다.', 0, 0),
    ('누구든지 이 법을 위반하는 사실을 발견하면 공정거래위원회에 신고할 수 있다.', 0, 0),
    ('협의회는 서면실태조사 결과를 공정하게 공개해야 한다.', 0, 0),
    ('조정조서는 재판상 화해와 동일한 효력을 가지므로 반드시 이행되어야 한다.', 0, 0),
    ('서면실태조사는 공정한 거래질서 확립을 위해 반드시 시행되어야 한다.', 0, 0),
    ('협의회의 운영 절차는 공정성을 보장할 수 있도록 대통령령에 따라야 한다.', 0, 0),
    ('공정거래위원회는 법에 따라 갑의 위반 혐의를 조사할 수 있다.', 0, 0),
    ('협의회의 조정 절차가 종료되면 시효는 새롭게 진행된다.', 0, 0),
]

for sentence, true_label, true_article in test_sentences:
    # 1️⃣ 불공정 여부 판별
    unfair_result = predict_unfair_clause(unfair_model, sentence)

    # 2️⃣ 불공정 문장일 경우 조항 예측 및 세부 조항 탐색
    if unfair_result["predicted_label"] == "불공정":
        predicted_article = predict_article(article_model, sentence)  # 예측된 조항 (숫자)
        law_details = find_most_similar_law_within_article(sentence, predicted_article)  # 법률 세부 조항 탐색
    else:
        predicted_article = 0  # 불공정이 아닐 경우 조항 없음
        law_details = {
            "Article number": None,
            "Article title": None,
            "Paragraph number": None,
            "Subparagraph number": None,
            "Article detail": None,
            "Paragraph detail": None,
            "Subparagraph detail": None
        }

    # 3️⃣ 결과 출력
    print(f"Sentence: {sentence}")
    print(f"Unfair: (Percent/label) = ({unfair_result['unfair_probability']}%/{'불공정' if true_label == 1 else '합법'})")
    print(f"Article: (Prediction/label) = ({predicted_article}/{true_article})")
    print(f"Article number: {law_details['Article number']}")
    print(f"Article title: {law_details['Article title']}")
    print(f"Article detail: {law_details['Article detail']}")
    print(f"Paragraph number: {law_details['Paragraph number']}")
    print(f"Paragraph detail: {law_details['Paragraph detail']}")
    print(f"Subparagraph number: {law_details['Subparagraph number']}")
    print(f"Subparagraph detail: {law_details['Subparagraph detail']}")
    print("-" * 50)


### 조 제목 추가