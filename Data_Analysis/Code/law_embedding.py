import pickle
import numpy as np
import json
from transformers import BertTokenizer, BertModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "klue/bert-base"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertModel.from_pretrained(MODEL_NAME).to(device)

with open("./Data_Analysis/Data/law_data.json", "r", encoding="utf-8") as f:
    law_data = json.load(f)

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



# ✅ 벡터 및 정보 저장
with open("./Data_Analysis/Data/law_embeddings.pkl", "wb") as f:
    pickle.dump({"law_embeddings": law_embeddings, "law_info": law_info}, f)

print("✅ 법률 벡터 데이터 저장 완료!")

# ✅ 저장된 벡터 데이터 불러오기
with open("./Data_Analysis/Data/law_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

law_embeddings = np.array(data["law_embeddings"])  # 벡터 데이터 복원
law_info = data["law_info"]  # 법률 정보 복원

print("✅ 법률 벡터 데이터 로드 완료!")
print(f"🔹 법률 벡터 개수: {len(law_embeddings)}")
print(f"🔹 법률 정보 예시: {law_info[:3]}")  # 일부 데이터 확인
