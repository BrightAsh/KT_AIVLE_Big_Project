import pandas as pd
import re

def process_contract_data(file_path):
    # 1. 텍스트 파일에서 데이터를 읽고 "제n조" 단위로 분리하고 빈 문자열을 제거하는 작업
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # "제n조" 단위로 텍스트를 분리
    pattern = r'(제\d+조(?!\S))'  # "제n조" 뒤에 공백이 있거나 끝났을 때
    matches = re.split(pattern, text)

    sentences = {}
    current_title = None
    sentence_number = {}

    for part in matches:
        if part.startswith('제') and '조' in part:
            current_title = re.sub(r'(\d+조)(\[)', r'\1 \2', part.strip())
            number_match = re.match(r'제(\d+)조', current_title)
            if number_match:
                num = number_match.group(1)
                if num in sentence_number:
                    sentence_number[num] += 1
                else:
                    sentence_number[num] = 1
                key = num if sentence_number[num] == 1 else f"{num}_{sentence_number[num]}"
            else:
                key = current_title

            sentences[key] = current_title.strip()
        elif current_title:
            if re.match(r'제\d+조\w', part.strip()):
                sentences[key] += part.strip()
            else:
                sentences[key] += part.strip()

    # 먼저 \n\n을 기준으로 분리
    def split_sentences(text):
        sentences = re.split(r'(\n\n)', text)

        result = []
        for sentence in sentences:
            if sentence.strip():  # 공백이 아닌 경우에만 처리
                result.extend(sentence.split("\n"))

        return result

    # "제n조"와 "제n조의m"을 그룹화하여 처리하는 함수
    def group_content_sections(data):
        grouped_data = {}
        for key, value in data.items():
            content_sentences = split_sentences(value.strip())  # content를 리스트화
            temp_key = key  # 기본 키는 현재 key로 설정 (예: "8")
            temp_content = []
            main_content = []  # "제n조"의 주요 내용을 유지하기 위한 리스트

            for sentence in content_sentences:
                match = re.match(r'제(\d+)조의(\d+) ?\[.*?\]', sentence)  # "제n조의m" 찾기
                if match:
                    # 이전 섹션 내용을 저장
                    if temp_key != key:  # 새로운 "제n조의m"이 시작될 때
                        grouped_data[temp_key] = temp_content
                        temp_content = []
                    # "제n조의m" 키 설정
                    num, sub_num = match.groups()
                    temp_key = f"{num}-{sub_num}"
                    temp_content.append(sentence)
                elif temp_key != key:  # "제n조의m" 안에 포함될 내용
                    temp_content.append(sentence)
                else:  # 기본 "제n조" 내용
                    main_content.append(sentence)

            # 마지막 섹션 저장
            if temp_key != key:
                grouped_data[temp_key] = temp_content
            grouped_data[key] = main_content  # 기본 "제n조" 내용 저장

        return grouped_data

    # 수정된 데이터 반환
    def modify_contract(data):
        grouped_data = group_content_sections(data)
        modified_data = {}
        for key, content in grouped_data.items():
            title_match = re.search(r'\[(.*?)\]', content[0]) if content else None
            title = title_match.group(1) if title_match else key

            # content[0] 삭제 (첫 번째 항목 제거)
            content = content[1:] if content else []

            modified_data[key] = {"title": title, "content": content}
        return modified_data

    # 정렬된 키에 맞게 데이터 반환
    def sort_grouped_data(grouped_data):
        # 조항 번호에 따라 정렬
        sorted_grouped_data = {}
        # 정렬 기준: 숫자와 텍스트를 모두 고려하여 정렬
        for key in sorted(grouped_data.keys(), key=lambda x: [int(i) if i.isdigit() else i for i in re.split(r'(\d+)', x)]):
            sorted_grouped_data[key] = grouped_data[key]
        return sorted_grouped_data

    # 빈 문자열을 제거하는 함수
    def del_empty_content(output_json):
        for key, value in output_json.items():
            # "content" 리스트가 존재할 경우
            if "content" in value:
                # "content" 리스트에서 빈 문자열 제거
                value["content"] = [item for item in value["content"] if item != ""]
        return output_json

    # 결과 데이터 수정
    modified_data = modify_contract(sentences)
    modified_data = sort_grouped_data(modified_data)
    # 빈 문자열을 제거
    modified_data = del_empty_content(modified_data)

    # 2. content 내용 수정 및 저장
    contents = {
        key: [
            sentence.strip().replace('"갑"', '갑').replace('"을"', '을')
            for sentence in value['content']
        ]
        for key, value in modified_data.items() if 'content' in value
    }
    contents = {key: sentences for key, sentences in contents.items() if sentences}

    # 3. DataFrame으로 변환
    rows = []
    for key, sentences in contents.items():
        combined_sentence = ' '.join(sentences)  # 문장들을 하나로 합치기
        title = modified_data[key]['title'] if key in modified_data else None
        rows.append({'key': key, 'sentence': combined_sentence, 'title': title})


    # DataFrame으로 변환
    df = pd.DataFrame(rows)

    # 4. 특수문자 기준으로 문장을 분리하는 함수
    def split_special_char_sentences(sentence):
        if re.search(r'(①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩|[①-⑩])', sentence):
            split_sentences = re.split(r'(①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩|[①-⑩])', sentence)
            sentences = [s.strip() for s in split_sentences if s.strip() and not re.match(r'[①-⑩]', s)]
            return sentences
        else:
            return [sentence.strip()]

    # 5. 번호가 포함된 문장을 찾는 함수
    def find_numbered_sentences(sentence):
        return re.findall(r'\d+\..+?(?=\d+\.|$)', sentence)  # '1.', '2.' 등으로 구분된 문장 찾기

    # 6. 특수문자 기준으로 문장을 분리한 후 번호 있는 문장을 리스트로 처리
    split_rows = []
    for _, row in df.iterrows():
        # ①, ②, ③ 등을 기준으로 문장을 분리
        sentences = split_special_char_sentences(row['sentence'])

        # 번호 기반 문장 리스트화
        sub_sentences = []
        for sentence in sentences:
            sub_sentence_list = find_numbered_sentences(sentence)
            sub_sentences.append(sub_sentence_list if sub_sentence_list else None)  # 빈 리스트는 None으로 처리

        # 리스트로 분리된 문장을 삭제하고 나머지 문장만 저장
        for i, sentence in enumerate(sentences):
            for numbered_sentence in sub_sentences[i] or []:  # None인 경우 빈 리스트로 처리:
                sentence = re.sub(re.escape(numbered_sentence), '', sentence).strip()

            split_rows.append({'key': row['key'], 'sentence': sentence, 'sub_sentence': sub_sentences[i],'title': row['title'] })

    # 7. 결과 DataFrame 생성
    split_df = pd.DataFrame(split_rows)

    return split_df


file_path = './Data_Analysis/Contract/24년 개정 직매입 표준거래계약서(면세점).txt'
# process_contract_data 함수를 실행하여 계약서 처리
result_df = process_contract_data(file_path)