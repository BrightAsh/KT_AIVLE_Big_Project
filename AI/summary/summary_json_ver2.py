from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import os, re, sys
sys.path.append(os.path.abspath("./Fast_Api"))
import modularization_ver1 as mo

def article_summary_AI(model, tokenizer, prompt, input_text, max_length=256):
    input_ids = tokenizer(f"{prompt}{input_text}", return_tensors="pt").input_ids
    summary_ids = model.generate(input_ids, max_length=max_length, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

prompt = (
    "다음은 계약서의 조항입니다. 이 조항의 주요 내용을 다음 기준에 따라 간략히 요약하세요:\n"
    "1. 이 조항이 규정하는 주요 목적 또는 대상\n"
    "2. 갑과 을의 권리와 의무\n"
    "3. 이행해야 할 절차와 조건\n"
    "4. 위반 시 결과 또는 조치\n\n"
    "요약은 각 기준에 따라 간결하고 명확하게 작성하며, 중복을 피하세요. "
    "조 제목과 관련된 핵심 정보를 반드시 포함하세요.\n\n"
)

model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
summary_results = []
txt = mo.hwp5txt_to_string('C:/Users/User/anaconda3/envs/bigp/Scripts/hwp5txt.exe', 'D:/KT_AIVLE_Big_Project/Data_Analysis/Contract/example.hwp')
articles = mo.contract_to_articles_ver2(txt)
for article_number, article_detail in articles.items():
    print('*'*50)
    match = re.match(r"(제\s?\d+조(?:의\s?\d+)?\s?)\[(.*?)\]\s?(.+)", article_detail, re.DOTALL)
    article_title = match.group(2)
    article_content = match.group(3)

    summary = article_summary_AI(model, tokenizer,prompt, article_detail)
    summary_results.append(
        {
            'article_number': article_number,  # 조 번호
            'article_title': article_title,  # 조 제목
            'summary': summary  # 조 요약
        }
    )
    print(f'{article_number}조 요약: {summary}')



# 모델 저장 경로 설정
save_directory = "D:/Model/article_summary_ver2/"

# 모델 및 토크나이저 저장
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
