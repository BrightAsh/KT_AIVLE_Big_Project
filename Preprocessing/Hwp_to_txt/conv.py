import os
import subprocess

# 🔹 변환할 한글 파일이 있는 폴더 또는 파일 경로
path = "example.hwp"  # 폴더 또는 파일 경로

# 🔹 hwp5txt 실행 파일 경로 (환경변수에 등록되어 있으면 'hwp5txt')
exefile = 'output'  # 한글 파일을 텍스트로 변환

# 🔹 한글 파일(.hwp)만 탐색하여 변환
def convert_hwp_to_txt(path):
    res = []

    # 파일인지 폴더인지 확인
    if os.path.isfile(path) and path.endswith('.hwp'):
        res.append(os.path.abspath(path))
    else:
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.hwp'):  # ✅ 한글 파일만 선택
                    filepath = os.path.join(root, file)
                    res.append(filepath)

    # 🔹 파일 변환
    for result in res:
        filename = result[:-4] + ".txt"  # ✅ .hwp → .txt
        output = '--output ' + '"' + filename + '"'
        result = '"' + result + '"'

        # 🔹 변환 명령어 확인
        command = f'{exefile} {output} {result}'
        print(f"변환 중: {command}")

        # 🔹 변환 실행 (보안 강화)
        subprocess.run(command, shell=True)

    print("✅ 변환이 완료되었습니다.")

# 🔹 변환 실행
convert_hwp_to_txt(path)
