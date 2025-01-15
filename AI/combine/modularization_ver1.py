################################################################################################
# 필요 패키지 import
################################################################################################
import pickle, openai, torch, json, os, numpy as np, torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
################################################################################################
# 토크나이징 모델 로드
################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "klue/bert-base"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertModel.from_pretrained(MODEL_NAME).to(device)

################################################################################################
# 모델 로드
################################################################################################
def load_trained_model_statice(model_class, model_file):
    model = model_class().to(device)
    state_dict = torch.load(model_file, map_location=device)
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    else:
        raise TypeError(f"모델 가중치 로드 실패: {model_file} (잘못된 데이터 타입 {type(state_dict)})")



################################################################################################
# 전체 모델과 데이터 로드
################################################################################################

def initialize_models(unfair_ver_sel, article_ver_sel, toxic_ver_sel):
    global unfair_model, article_model, toxic_model, toxic_tokenizer, law_data, law_embeddings
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
    class BertArticleClassifier(nn.Module):
        def __init__(self, bert_model_name="klue/bert-base", hidden_size=256, num_classes=27):
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
    # 불공정 조항 판별 모델 로드
    unfair_model = load_trained_model_statice(BertMLPClassifier, f"./Data_Analysis/Model/{unfair_ver_sel}/klue_bert_mlp.pth")

    # 조항 예측 모델 로드
    article_model = load_trained_model_statice(BertArticleClassifier, f"./Data_Analysis/Model/{article_ver_sel}/klue_bert_mlp.pth")

    # 독소 조항 판별 모델 로드
    toxic_model = BertForSequenceClassification.from_pretrained(f"./Data_Analysis/Model/{toxic_ver_sel}/").to(device)
    toxic_tokenizer = BertTokenizer.from_pretrained(f"./Data_Analysis/Model/{toxic_ver_sel}/")

    # 법률 데이터 로드
    with open("./Data_Analysis/Data/law_embeddings.pkl", "rb") as f:
        data = pickle.load(f)
    with open("./Data_Analysis/Data/law_data_ver2.json", "r", encoding="utf-8") as f:
        law_data = json.load(f)
    law_embeddings = np.array(data["law_embeddings"])

    print("✅ 모든 모델 및 데이터 로드 완료!")
################################################################################################
# 불공정 식별
################################################################################################
def predict_unfair_clause(model, sentence, threshold=0.5):
    model.eval()
    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(inputs["input_ids"], inputs["attention_mask"])
        unfair_prob = output.item()
    return 1 if unfair_prob >= threshold else 0
################################################################################################
# 조항 예측
################################################################################################
def predict_article(model,sentence):
    model.eval()
    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(inputs["input_ids"], inputs["attention_mask"])
        predicted_idx = torch.argmax(output).item()
        predicted_article = predicted_idx + 4
    return predicted_article
################################################################################################
# 독소 식별
################################################################################################
def predict_toxic_clause(toxic_model, toxic_tokenizer, sentence, threshold=0.5):
    """독소 조항 여부 예측"""
    toxic_model.eval()
    inputs = toxic_tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        output = toxic_model(**inputs).logits
        if output.shape[1] == 1:
            toxic_prob = torch.sigmoid(output).cpu().numpy()[0, 0]  # 단일 확률값
        else:
            toxic_prob = torch.softmax(output, dim=1).cpu().numpy()[0, 1]  # 독소 조항(1) 확률
    return 1 if toxic_prob >= threshold else 0
################################################################################################
# 코사인 유사도
################################################################################################
def find_most_similar_law_within_article(sentence, predicted_article, law_data):
    contract_embedding = bert_model(**tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)).pooler_output.cpu().detach().numpy()[0]
    predicted_article = str(predicted_article)
    matching_article = []
    for article in law_data:
        if article["article_number"].split()[1].startswith(predicted_article):
            matching_article.append(article)
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
    best_match = None
    best_similarity = -1
    for article in matching_article:
        article_title = article.get("article_title", None)
        article_detail = article.get("article_content", None)
        if article_title:
            article_embedding = bert_model(**tokenizer(article_title, return_tensors="pt", padding=True, truncation=True).to(device)).pooler_output.cpu().detach().numpy()[0]
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
        for clause in article.get("clauses", []):
            clause_text = clause["content"].strip()
            clause_number = clause["clause_number"]
            if clause_text:
                clause_embedding = bert_model(**tokenizer(clause_text, return_tensors="pt", padding=True, truncation=True).to(device)).pooler_output.cpu().detach().numpy()[0]
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
        if best_match and best_match["subparagraphs"]:
            for subclause in best_match["subparagraphs"]:
                if isinstance(subclause, str):
                    subclause_embedding = bert_model(**tokenizer(subclause, return_tensors="pt", padding=True, truncation=True).to(device)).pooler_output.cpu().detach().numpy()[0]
                    similarity = cosine_similarity([contract_embedding], [subclause_embedding])[0][0]
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match["subparagraph_detail"] = subclause
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
################################################################################################
# 설명 AI
################################################################################################
def explanation_AI(sentence, unfair_label, toxic_label, law=None):
    with open('./key/openAI_key.txt', 'r') as file:
        openai.api_key = file.readline().strip()
    os.environ['OPENAI_API_KEY'] = openai.api_key
    client = openai.OpenAI()
    if unfair_label == 0 and toxic_label == 0:
        return None
    prompt = f"""
        아래 계약 조항이 특정 법률을 위반하는지 분석하고, 조항(제n조), 항(제m항), 호(제z호) 형식으로 **명확하고 간결하게** 설명하세요.
        📌 **설명할 때는 사용자에게 직접 말하는 듯한 자연스러운 문장으로 구성하세요.**
        📌 **한눈에 보기 쉽도록 짧고 명확한 문장을 사용하세요.**
        📌 **불공정 라벨이 1인 경우에는 불공정에 관한 설명만 하고, 독소 라벨이 1인 경우에는 독소에 관한 설명한 하세요**

        계약 조항: "{sentence}"
        불공정 라벨: {unfair_label} (1일 경우 불공정)
        독소 라벨: {toxic_label} (1일 경우 독소)   
        {f"관련 법 조항: {law}" if law else "관련 법 조항 없음"}

        🔴 **불공정 조항일 경우:**
        1️⃣ **위반된 법 조항을 '제n조 제m항 제z호' 형식으로 먼저 말해주세요.**
        2️⃣ **위반 이유를 간결하게 설명하세요.**
        3️⃣ **설명은 '🚨 법 위반!', '🔍 이유' 순서로 구성하세요.**

        ⚫ **독소 조항일 경우:**
        1️⃣ **법 위반이 아니라면, 해당 조항이 계약 당사자에게 어떤 위험을 초래하는지 설명하세요.**
        2️⃣ **구체적인 문제점을 짧고 명확한 문장으로 설명하세요.**
        3️⃣ **설명은 '💀 독소 조항', '🔍 이유' 순서로 구성하세요.**

        ⚠️ 참고: 제공된 법 조항이 실제로 위반된 조항이 아닐 경우, **GPT가 판단한 적절한 법 조항을 직접 사용하여 설명하세요.** 
        그러나 원래 제공된 법 조항과 비교하여 반박하는 방식으로 설명하지 마세요.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content":
                            "당신은 계약서 조항이 특정 법률을 위반하는지 분석하는 법률 전문가입니다. \
                            불공정 조항의 경우, 어떤 법 조항을 위반했는지 조항(제n조), 항(제m항), 호(제z호) 형식으로 정확히 명시한 후 설명하세요. \
                            만약 제공된 법 조항이 실제로 위반된 조항이 아니라면, GPT가 판단한 적절한 법 조항을 사용하여 설명하세요. \
                            독소 조항은 법률 위반이 아니라 계약 당사자에게 미치는 위험성을 중심으로 설명하세요."
                   },
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    ).choices[0].message.content
    return response
################################################################################################
# 파이프 라인
################################################################################################
def pipline(sentence):
    print(f'Input: {sentence}')
    unfair_result = predict_unfair_clause(unfair_model, sentence)
    if unfair_result:
        predicted_article = predict_article(article_model, sentence)  # 예측된 조항
        law_details = find_most_similar_law_within_article(sentence, predicted_article, law_data)
        toxic_result = 0
    else:
        toxic_result = predict_toxic_clause(toxic_model, toxic_tokenizer, sentence)
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
    law_text = []
    if law_details.get("Article number"):
        law_text.append(f"{law_details['Article number']}({law_details['Article title']})")
    if law_details.get("Article detail"):
        law_text.append(f": {law_details['Article detail']}")
    if law_details.get("Paragraph number"):
        law_text.append(f" {law_details['Paragraph number']}: {law_details['Paragraph detail']}")
    if law_details.get("Subparagraph number"):
        law_text.append(f" {law_details['Subparagraph number']}: {law_details['Subparagraph detail']}")
    law = "".join(law_text) if law_text else None

    explain = explanation_AI(sentence, unfair_result, toxic_result, law)

    result = {
        'Sentence': sentence,
        'Unfair': unfair_result,
        'Toxic': toxic_result,
        'law': law,
        'explain': explain
    }
    return result