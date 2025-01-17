import os
import sys
import time

sys.path.append(os.path.abspath("./AI/combine"))
import modularization_ver1 as mo

# 계약서 이름 설정
contract_name = 'example.hwp'

# 모델 초기화
mo.initialize_models()

# 실행 시간 측정
start_time = time.time()  # 시작 시간 기록
indentification_results, summary_results = mo.pipline(contract_name)
end_time = time.time()  # 종료 시간 기록

# 실행 시간 출력
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")
