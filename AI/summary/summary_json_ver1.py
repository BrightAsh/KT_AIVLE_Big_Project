from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import json, os, sys
sys.path.append(os.path.abspath("./AI/separate"))
from AI.separate import separate_module_ver1 as sp

def generate_summaries_for_all(data):
    model = AutoModelForSeq2SeqLM.from_pretrained('eenzeenee/t5-base-korean-summarization')
    tokenizer = AutoTokenizer.from_pretrained('eenzeenee/t5-base-korean-summarization')
    prefix = "summarize: "
    
    summaries = {}

    # 모든 키에 대해 요약을 생성
    for key in data.index:  # 데이터프레임의 모든 행을 순회
        # 'sentence'와 'sub_sentence'를 결합하여 입력값 생성
        sentence = str(data.loc[key, 'sentence'])
        sub_sentence = str(data.loc[key, 'sub_sentence'])
        
        # 두 문장을 '/'로 구분하여 연결
        input_text = sentence + " / " + sub_sentence  # 구분자를 '/'로 설정

        # 모델 입력 준비
        inputs = [prefix + input_text]

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
summary_result = sp.generate_summaries_for_all(separate_json)

print(summary_result)