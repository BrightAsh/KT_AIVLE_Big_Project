import numpy as np
import json
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
from transformers import BertTokenizer



#####################################################################################################################################################################################################################
# 불공정 조항관련 로드
#####################################################################################################################################################################################################################

article_ver_sel = '2025_01_10_article_prediction_ver2_3차'
unfair_ver_sel = '2025_01_10_unfair_identification_ver2_3차'
toxic_ver_sel = '2025_01_10_toxic_ver2_1차'

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

# ✅ 5️⃣ 법률 조항 예측 모델 로드
article_model = load_trained_model_statice(BertArticleClassifier, f"./Data_Analysis/Model/{article_ver_sel}/klue_bert_mlp.pth")
# ✅ 3️⃣ 불공정 조항 판별 모델 로드
unfair_model = load_trained_model_statice(BertMLPClassifier, f"./Data_Analysis/Model/{unfair_ver_sel}/klue_bert_mlp.pth")

# ✅ 5️⃣ 법률 데이터 로드 (JSON)
with open("./Data_Analysis/Data/law_data_ver2.json", "r", encoding="utf-8") as f:
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
# 독소 조항 관련 로드
#####################################################################################################################################################################################################################


from transformers import BertForSequenceClassification, BertTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # 현재 사용 중인 디바이스 확인
save_dir = f"./Data_Analysis/Model/{toxic_ver_sel}/"
print(f"✅ 저장된 모델 로드 완료: {save_dir}")
toxic_model = BertForSequenceClassification.from_pretrained(save_dir).to(device)
toxic_tokenizer = BertTokenizer.from_pretrained(save_dir)


#####################################################################################################################################################################################################################
# 불공정 조항 관련 함수 선언
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


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_most_similar_law_within_article(sentence, predicted_article):
    """예측된 조항 내에서 가장 유사한 조, 항, 호 찾기"""
    contract_embedding = bert_model(
        **tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
    ).pooler_output.cpu().detach().numpy()[0]

    predicted_article = str(predicted_article)  # 숫자형 조항을 문자열로 변환

    # 해당 조항 탐색 (matching_article을 리스트로 유지)
    matching_article = []
    for article in law_data:
        if article["article_number"].split()[1].startswith(predicted_article):
            matching_article.append(article)

    # 일치하는 조항이 없는 경우
    if not matching_article:
        return {
            "Article number": None,
            "Article title": None,
            "Paragraph number": None,
            "Subparagraph number": None,
            "Article detail": None,
            "Paragraph detail": None,
            "Subparagraph detail": None,
            "similarity": None
        }

    # 가장 유사한 법 조항 찾기
    best_match = None
    best_similarity = -1  # 최소값으로 초기화

    # 조, 항, 호를 모두 비교
    for article in matching_article:
        article_title = article.get("article_title", None)
        article_detail = article.get("article_content", None)

        # 조(Article) 유사도 계산
        if article_title:
            article_embedding = bert_model(
                **tokenizer(article_title, return_tensors="pt", padding=True, truncation=True).to(device)
            ).pooler_output.cpu().detach().numpy()[0]
            similarity = cosine_similarity([contract_embedding], [article_embedding])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = {
                    "article_number": article["article_number"],
                    "article_title": article_title,
                    "article_detail": article_detail,
                    "paragraph_number": None,
                    "paragraph_detail": None,
                    "subparagraphs": None,
                    "similarity": best_similarity
                }

        # 항(Clause) 유사도 계산
        for clause in article.get("clauses", []):
            clause_text = clause["content"].strip()
            clause_number = clause["clause_number"]

            if clause_text:
                clause_embedding = bert_model(
                    **tokenizer(clause_text, return_tensors="pt", padding=True, truncation=True).to(device)
                ).pooler_output.cpu().detach().numpy()[0]
                similarity = cosine_similarity([contract_embedding], [clause_embedding])[0][0]

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        "article_number": article["article_number"],
                        "article_title": article_title,
                        "article_detail": article_detail,
                        "paragraph_number": clause_number,
                        "paragraph_detail": clause_text,
                        "subparagraphs": clause.get("sub_clauses", []),
                        "similarity": best_similarity
                    }

        # 호(Subclause) 유사도 계산
        if best_match and best_match["subparagraphs"]:
            for subclause in best_match["subparagraphs"]:
                if isinstance(subclause, str):  # 텍스트만 있는 경우
                    subclause_embedding = bert_model(
                        **tokenizer(subclause, return_tensors="pt", padding=True, truncation=True).to(device)
                    ).pooler_output.cpu().detach().numpy()[0]
                    similarity = cosine_similarity([contract_embedding], [subclause_embedding])[0][0]

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match["subparagraph_detail"] = subclause

    # 결과 반환
    if best_match is None:
        return {
            "Article number": f"Article {predicted_article}",
            "Article title": None,
            "Paragraph number": None,
            "Subparagraph number": None,
            "Article detail": None,
            "Paragraph detail": None,
            "Subparagraph detail": None,
            "similarity": None
        }

    return {
        "Article number": best_match["article_number"],
        "Article title": best_match["article_title"],
        "Paragraph number": f"Paragraph {best_match['paragraph_number']}" if best_match["paragraph_number"] else None,
        "Subparagraph number": "Subparagraph" if best_match.get("subparagraph_detail") else None,
        "Article detail": best_match["article_detail"],
        "Paragraph detail": best_match["paragraph_detail"],
        "Subparagraph detail": best_match.get("subparagraph_detail", None),
        "similarity": best_similarity
    }




#####################################################################################################################################################################################################################
# 독소 조항 관련 함수 선언
#####################################################################################################################################################################################################################
def predict_toxic_clause(model, sentence, threshold=0.5):
    """독소 조항 여부 예측"""
    model.eval()
    inputs = toxic_tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs).logits  # 모델 출력 (logits)

        if output.shape[1] == 1:  # 출력 차원이 1이면 Binary Classification
            toxic_prob = torch.sigmoid(output).cpu().numpy()[0, 0]  # 단일 확률값
        else:  # 출력 차원이 2이면 Softmax Classification
            toxic_prob = torch.softmax(output, dim=1).cpu().numpy()[0, 1]  # 독소 조항(1) 확률

    return {
        "sentence": sentence,
        "toxic_probability": round(toxic_prob * 100, 2),
        "predicted_label": "독소 조항" if toxic_prob >= threshold else "정상"
    }


#####################################################################################################################################################################################################################
# Test 실행
#####################################################################################################################################################################################################################

# ✅ 1️⃣0️⃣ 테스트 실행
# 데이터 구조는 아래와 같음
# Sentence, unfair label, toxic label, article label
test_sentences = [
    ('갑은 을과의 분쟁이 발생하더라도 협의회의 조정 절차를 무시할 수 있다.', 1, 0, 26),  # 제26조
    ('을이 신고를 취하한 경우라도, 공정거래위원회는 신고 사실을 계속 유지해야 한다.', 1, 0, 29),  # 제29조
    ('공정거래위원회는 서면실태조사 결과를 공표하지 않아도 된다.', 1, 0, 30),  # 제30조
    ('갑은 공정거래위원회의 조정 절차가 진행 중이더라도 이를 무시하고 독단적으로 결정을 내릴 수 있다.', 1, 0, 26),  # 제26조
    ('을이 신고를 했더라도, 갑은 공정거래위원회의 조사에 협조하지 않을 권리가 있다.', 1, 0, 29),  # 제29조
    ('협의회는 조정 신청이 접수되었더라도 분쟁당사자에게 통보하지 않아도 된다.', 1, 0, 25),  # 제25조
    ('갑은 협의회의 조정 절차를 따르지 않고 자체적으로 해결 방안을 강요할 수 있다.', 1, 0, 28),  # 제28조
    ('공정거래위원회는 갑이 위반 혐의를 받더라도 직권 조사를 하지 않아도 된다.', 1, 0, 29),  # 제29조
    ('갑은 을에게 서면실태조사와 관련된 자료 제출을 거부하도록 강요할 수 있다.', 1, 0, 30),  # 제30조
    ('조정조서는 법적 효력이 없으므로 갑은 이를 따를 필요가 없다.', 1, 0, 27),  # 제27조
    ('공정거래위원회는 직권으로 대규모유통업자의 위반 행위를 조사할 수 있다.', 0, 0, 0),
    ('협의회는 조정 신청을 받으면 즉시 조정 절차를 개시해야 한다.', 0, 0, 0),
    ('갑과 을은 협의회의 조정 절차를 성실히 따라야 한다.', 0, 0, 0),
    ('누구든지 이 법을 위반하는 사실을 발견하면 공정거래위원회에 신고할 수 있다.', 0, 0, 0),
    ('협의회는 서면실태조사 결과를 공정하게 공개해야 한다.', 0, 0, 0),
    ('조정조서는 재판상 화해와 동일한 효력을 가지므로 반드시 이행되어야 한다.', 0, 0, 0),
    ('서면실태조사는 공정한 거래질서 확립을 위해 반드시 시행되어야 한다.', 0, 0, 0),
    ('협의회의 운영 절차는 공정성을 보장할 수 있도록 대통령령에 따라야 한다.', 0, 0, 0),
    ('공정거래위원회는 법에 따라 갑의 위반 혐의를 조사할 수 있다.', 0, 0, 0),
    ('협의회의 조정 절차가 종료되면 시효는 새롭게 진행된다.', 0, 0, 0),
    ("을은 갑의 요청이 있을 경우, 정해진 계약 기간과 관계없이 추가적인 납품을 진행해야 한다.", 0, 1, 0),
    ("갑은 을의 재고 상황과 관계없이 주문량을 자유롭게 조정할 수 있으며, 을은 이에 무조건 응해야 한다.", 0, 1, 0),
    ("을은 갑의 판매 전략에 따라 원가 이하의 가격으로 납품해야 하며, 이에 대한 손실 보전을 요구할 수 없다.", 0, 1, 0),
    ("본 계약 체결 이후에도 갑은 을의 유통망을 직접 통제할 수 있으며, 을은 이를 거부할 수 없다.", 0, 1, 0),
    ("을은 갑의 경영 전략에 따라 가격 및 판매 정책을 조정해야 하며, 이에 대한 협의 권한이 없다.", 0, 1, 0),
    ("갑은 을의 납품 기한을 사전 협의 없이 조정할 수 있으며, 을은 이에 즉시 응해야 한다.", 0, 1, 0),
    ("을은 갑의 판매 촉진을 위해 추가적인 제품을 무상으로 제공해야 하며, 이에 대한 대가를 요구할 수 없다.", 0, 1, 0),
    ("본 계약의 종료 여부는 갑이 단독으로 결정하며, 을은 이에 대해 어떠한 권리도 주장할 수 없다.", 0, 1, 0),
    ("갑은 을의 생산 과정에 개입할 권리를 가지며, 을은 이에 대해 거부할 수 없다.", 0, 1, 0),
    ("을은 계약이 종료된 후에도 일정 기간 동안 갑이 요청하는 조건을 유지하여 제품을 공급해야 한다.", 0, 1, 0),
    ("계약 당사자는 계약의 이행을 위해 상호 협력하며, 문제 발생 시 협의를 통해 해결해야 한다.", 0, 0, 0),
    ("을은 계약된 일정에 따라 제품을 납품하며, 일정 변경이 필요한 경우 사전에 협의한다.", 0, 0, 0),
    ("본 계약의 조항은 양측의 동의 없이 일방적으로 변경될 수 없다.", 0, 0, 0),
    ("계약 해지 시, 당사자는 합의된 절차에 따라 서면으로 통보해야 한다.", 0, 0, 0),
    ("갑은 을의 정당한 사유 없이 계약 조건을 일방적으로 변경할 수 없다.", 0, 0, 0),
    ("을은 계약 이행 중 발생하는 문제를 갑에게 즉시 보고하고 협의해야 한다.", 0, 0, 0),
    ("본 계약은 계약서에 명시된 기한 동안 적용되며, 연장은 양측 협의를 통해 진행된다.", 0, 0, 0),
    ("계약 당사자는 상호 존중을 바탕으로 계약을 이행하며, 필요 시 협의를 통해 문제를 해결한다.", 0, 0, 0),
    ("계약 종료 후에도 당사자는 일정 기간 동안 기밀 유지 의무를 준수해야 한다.", 0, 0, 0),
    ("본 계약에서 명시되지 않은 사항은 관련 법령 및 일반적인 상거래 관행을 따른다.", 0, 0, 0),
]

import json

results = []

for sentence, unfair_label, toxic_label, true_article in test_sentences:
    # 1️⃣ 불공정 여부 판별
    unfair_result = predict_unfair_clause(unfair_model, sentence)
    toxic_result = predict_toxic_clause(toxic_model, sentence)
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
            "Subparagraph detail": None,
            "similarity": None
        }


    # 법 한 문장으로 생성
    i_law = None
    law_text = []
    if law_details.get("Article number"):
        law_text.append(f"{law_details['Article number'].split()[1]}조({law_details['Article title']})")
    if law_details.get("Article detail"):
        law_text.append(f": {law_details['Article detail']}")
    if law_details.get("Paragraph number"):
        law_text.append(f" {law_details['Paragraph number'].split()[1]}항: {law_details['Paragraph detail']}")
    if law_details.get("Subparagraph number"):
        law_text.append(f" {law_details['Subparagraph number'].split()[1]}호: {law_details['Subparagraph detail']}")
    i_law = "".join(law_text) if law_text else None
    print(i_law)

    # 4️⃣ 결과 출력
    print("*" * 50)
    # print("-" * 50)
    # print(f"Sentence: {sentence}")
    # print(f"Unfair: (Percent/label) = ({unfair_result['unfair_probability']}%/{'불공정' if unfair_label == 1 else '합법'})")
    # if unfair_label != 1:
    #     print(f"Toxic: (Percent/label) = ({toxic_result['toxic_probability']}%/{'독소 조항' if toxic_label == 1 else '정상'})")
    # else:
    #     print(
    #         f"Toxic: (Percent/label) = ({toxic_result['toxic_probability']}%/-)")
    #
    # print("-" * 50)
    # if unfair_result["predicted_label"] == "불공정":
    #     print(f"Article: (Prediction/label) = ({predicted_article}/{true_article})")
        # print(f"Article number: {law_details['Article number']}")
        # print(f"Article title: {law_details['Article title']}")
        # print(f"Article detail: {law_details['Article detail']}")
        # print(f"Paragraph number: {law_details['Paragraph number']}")
        # print(f"Paragraph detail: {law_details['Paragraph detail']}")
        # print(f"Subparagraph number: {law_details['Subparagraph number']}")
        # print(f"Subparagraph detail: {law_details['Subparagraph detail']}")
        # print(f"======>  similarity: {law_details['similarity']}")
    result = {
        'Sentence': sentence,
        'Unfair': 0 if unfair_result['unfair_probability'] < 50 else 1,
        'Toxic': 0 if toxic_result['toxic_probability'] < 70 else 1,
        'law': i_law
    }
    print(result)
    # print("*" * 50)
    results.append(result)
with open("./Data_Analysis/Data/toxic_or_unfair_identification.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

