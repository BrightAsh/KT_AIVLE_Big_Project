import os
import sys
import time

# import modularization_v1 as mo
import modularization_v2 as mo

# 계약서 이름 설정
contract_path ='D:/KT_AIVLE_Big_Project/Data_Analysis/Contract/example.hwp'
# 모델 초기화
mo.initialize_models()

# 실행 시간 측정
start_time = time.time()  # 시작 시간 기록
indentification_results, summary_results = mo.pipline(contract_path)
end_time = time.time()  # 종료 시간 기록

# 실행 시간 출력
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")
