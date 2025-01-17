from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import json, os, sys
import torch

# 상대 경로로 경로 추가
sys.path.append(os.path.abspath("./Preprocessing/separate"))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# # 모듈 import
import contract_to_articles as sp

model = AutoModelForSeq2SeqLM.from_pretrained('eenzeenee/t5-base-korean-summarization')
tokenizer = AutoTokenizer.from_pretrained('eenzeenee/t5-base-korean-summarization')
name = 'summary_ver1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def generate_summaries_for_all(data):

    prefix = "summarize: "
    
    summaries = {}

    # 모든 키에 대해 요약을 생성
    for key in data:  # 데이터프레임의 모든 행을 순회

        sample = data[key]
        inputs = [prefix + sample]
        
        # 토크나이징 및 모델 입력
        inputs = tokenizer(inputs, max_length=3000, truncation=True, return_tensors="pt")

        # 모델을 통한 요약 생성
        output = model.generate(**inputs, num_beams=5, do_sample=True, min_length=100, max_length=300, temperature=1.5)

        # 디코딩 및 첫 번째 문장 추출
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        result = nltk.sent_tokenize(decoded_output.strip())[0]

        # 요약 결과 저장
        summaries[key] = result

    return summaries

separate_json = sp.separate_json
summary_result = generate_summaries_for_all(separate_json)

save_dir = f"./Data_Analysis/Model/{name}/"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# ✅ 모델 로드 함수
def load_trained_model(model_path):
    return AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

# ✅ 모델 로드 실행
loaded_model = load_trained_model(save_dir)
loaded_tokenizer = AutoTokenizer.from_pretrained(save_dir)  # ✅ 토크나이저도 같은 경로에서 로드