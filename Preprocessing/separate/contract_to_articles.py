## 신버전 ### 분리됨

import re

def extract_and_modify_contract(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # "제n조" 단위로 텍스트를 분리
    pattern = r'(제\d+조(?!\S))'  # "제n조" 뒤에 공백이 있거나 끝났을 때
    matches = re.split(pattern, text)

    data = {}
    section_counter = {}  # 각 "제n조"의 중복 횟수를 추적하기 위한 딕셔너리
    for i in range(1, len(matches), 2):
        section_title = matches[i].strip()
        section_content = matches[i + 1].strip()

        # "제n조" 번호 추출
        section_num = re.match(r'제(\d+)조', section_title).groups()[0]

        # 중복 처리
        if section_num in data:
            if section_num in section_counter:
                section_counter[section_num] += 1
            else:
                section_counter[section_num] = 2
            new_title = f"{section_num}_{section_counter[section_num]}"
        else:
            section_counter[section_num] = 1
            new_title = section_num

        data[new_title] = section_content

    def split_sentences(text):
        # 문장을 분리만 수행
        return re.split(r'(\n\n)', text)

    # "제n조"와 "제n조의m"을 그룹화하여 처리하는 함수
    def group_content_sections(data):
        grouped_data = {}

        temp_content = {}  # 세부 항목들 임시 저장

        for key, value in data.items():
            content_sentences = split_sentences(value.strip())  # 문장 분리 수행

            clean_value = re.sub(r'\n\n', '', value.strip())
            clean_value = re.sub(r'\"갑\"', '갑', clean_value)  # \"갑\"을 갑으로 변환
            clean_value = re.sub(r'\"을\"', '을', clean_value)  # \"을\"을 을로 변환
            clean_value = re.sub(r'\\"([^"]+)\\"', r"'\1'", clean_value)
            clean_value = re.sub(r'\"([^"]+)\"', r"'\1'", clean_value)

            # "제n조" 부분을 n으로만 추출하여 저장
            grouped_data[key] = [f"제{key}조 {clean_value}"]

            # "제n조의m" 형식 처리
            temp_key = None
            for sentence in content_sentences:
                sentence = re.sub(r'\n\n', '', sentence.strip())
                sentence = re.sub(r'\"갑\"', '갑', sentence)
                sentence = re.sub(r'\"을\"', '을', sentence)
                sentence = re.sub(r'\\"([^"]+)\\"', r"'\1'", sentence)
                sentence = re.sub(r'\"([^"]+)\"', r"'\1'", sentence)
                match_sub_section = re.match(r'제(\d+)조의(\d+)', sentence)  # "제n조의m" 찾기
                if match_sub_section:
                    # 세부 항목 처리
                    num, sub_num = match_sub_section.groups()
                    temp_key = f"{num}-{sub_num}"
                    if temp_key not in temp_content:
                        temp_content[temp_key] = []
                    temp_content[temp_key].append(sentence.strip())
                    # 추가된 것
                    grouped_data[num] = [s.split(sentence.strip())[0] if sentence.strip() in s else s for s in grouped_data[num]]

                else:
                    match_section = re.match(r'제(\d+)조', sentence)  # "제n조" 구분
                    if match_section:
                        num = match_section.groups()[0]
                        temp_key = f"{num}"
                        if temp_key not in temp_content:
                            temp_content[temp_key] = []
                    if temp_key is not None:
                        temp_content[temp_key].append(sentence.strip())

        # 세부 항목들을 각 조문 바로 뒤에 올 수 있도록 조정
        for key, value in temp_content.items():
            if key in grouped_data:
                grouped_data[key].extend(value)
            else:
                grouped_data[key] = value

        return grouped_data

    def sort_grouped_data(grouped_data):
        # 조항 번호에 따라 정렬
        sorted_grouped_data = {}
        # 정렬 기준: 숫자와 텍스트를 모두 고려하여 정렬
        for key in sorted(grouped_data.keys(), key=lambda x: [int(i) if i.isdigit() else i for i in re.split(r'(\d+)', x)]):
            sorted_grouped_data[key] = grouped_data[key]
        return sorted_grouped_data

    def del_empty_content(output_json):
        for key, value in output_json.items():
            if isinstance(value, list):
                output_json[key] = [item for item in value if item]
        return output_json

    def merge_sentences(grouped_data):
        for key, value in grouped_data.items():
            grouped_data[key] = ' '.join(value)  # 리스트 내부 문장을 하나로 합침
        return grouped_data

    grouped_data = group_content_sections(data)
    grouped_data = sort_grouped_data(grouped_data)
    grouped_data = del_empty_content(grouped_data)
    grouped_data = merge_sentences(grouped_data)

    return grouped_data