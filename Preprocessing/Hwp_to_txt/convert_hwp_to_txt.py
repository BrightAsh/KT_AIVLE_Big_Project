#외부 프로그램이나 명령어를 실행하기 위한 모듈
import subprocess
#파일 및 디렉토리 경로를 다루기 위한 기본 모듈
import os

#출력파일 경로가 지정되지 않는다면 입력파일 디렉토리에 출력파일 저장 
def hwp5txt_to_txt(hwp_path, output_dir=None):
    if not os.path.exists(hwp_path):
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {hwp_path}")

    if output_dir is None:
        output_dir = os.path.dirname(hwp_path)
    
    base_name = os.path.splitext(os.path.basename(hwp_path))[0]
    txt_file_path = os.path.join(output_dir, f"{base_name}.txt")

    # hwp5txt 명령어 실행
    command = f"hwp5txt \"{hwp_path}\" > \"{txt_file_path}\""
    subprocess.run(command, shell=True, check=True)

    print(f"텍스트 파일로 저장 완료: {txt_file_path}")
    return txt_file_path


# 사용 예시
#hwp5txt_to_txt("example.hwp")

def hwp5txt_to_string(hwp_path):
    if not os.path.exists(hwp_path):
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {hwp_path}")

    # hwp5txt 명령어 실행 (출력 결과를 파이썬 문자열로 받기)
    command = f"hwp5txt \"{hwp_path}\""
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        encoding='utf-8',  # 🔥 인코딩을 utf-8로 변경
        errors='ignore'    # 🔥 인코딩 오류 무시
    )

    # 추출된 텍스트를 문자열로 반환
    extracted_text = result.stdout

    print("텍스트 추출이 완료되었습니다.")
    return extracted_text

# 사용 예시
text_content = hwp5txt_to_string("example.hwp")

print(text_content)
