from fastapi import FastAPI, UploadFile, File, HTTPException
#from AI.combine.modularization_ver1 import *
import os ,sys
import shutil
import uuid
import modularization_v1 as mo
app = FastAPI()

# 임시 폴더 경로
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)


sys.path.append(os.path.abspath("./AI/combine"))


#파일 업로드 및 텍스트 변환
#POST /api/upload → HWP 파일 업로드 및 텍스트 변환
@app.post("/api/upload/")
async def upload_file(file: UploadFile = File(...)):
    # 업로드된 파일이 HWP인지 확인
    if not file.filename.endswith('.hwp'):
        raise HTTPException(status_code=400, detail="HWP 파일만 업로드할 수 있습니다.")

    # 임시 파일 경로 생성
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(TEMP_DIR, unique_filename)

    # 업로드된 파일 임시저장
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        print("파일이 저장되었습니다.")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    hwp5txt_exe = os.path.join(current_dir,'Data_Analysis', 'Contract', 'hwp5txt.exe')
    #converted_file = hwp5txt_to_string(hwp5txt_exe, file_path)

    mo.initialize_models()
    indentification_results, summary_results = mo.pipline(file_path)
    # 임시 저장된 파일 삭제
    os.remove(file_path)

    return {"filename": indentification_results, "content": summary_results}



#요약 요청
#POST /api/summarize → 변환된 텍스트 요약
#@app.post("/upload/")

'''
return article_number: int
    # clause_number : int
    # subclause_number: int
    article_content: str

'''

#독소조항 판별 요청
#POST /api/check-unfair-clauses → 변환된 텍스트 독소조항 판별
#@app.post("/upload/")

'''
return {article_number: int
    clause_number : int
    subclause_number: int                        
    Unfair: str
    Toxic: str
    explain: str}
''' 

#전체 처리 (필요한 경우 통합 요청, 선택적 제공)
#POST /api/full-analysis → 파일 업로드부터 요약 및 독소조항 판별까지 한 번에 처리


