import openai
import json
import os

# OpenAI API Key 설정
with open('./key/openAI_key.txt', 'r') as file:
    openai.api_key = file.readline().strip()
os.environ['OPENAI_API_KEY'] = openai.api_key
client = openai.OpenAI()

def load_json(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)

def get_explanation(sentence, unfair_label, toxic_label, law=None):
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

    # OpenAI API 호출
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


def process_data(input_file, output_file):
    data = load_json(input_file)
    for item in data:
        print(item)
        item["explain"] = get_explanation(
            item["Sentence"], item["Unfair"], item["Toxic"], item["law"]
        )
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"✅ 결과가 {output_file} 파일에 저장되었습니다.")


# 실행 (입력 JSON → 설명 생성 → 출력 JSON 저장)
input_json_file = "./Data_Analysis/Data/toxic_or_unfair_identification.json"  # 입력 JSON 파일명
output_json_file = "./Data_Analysis/Data/identification_explain.json"  # 결과 저장 파일명
process_data(input_json_file, output_json_file)
