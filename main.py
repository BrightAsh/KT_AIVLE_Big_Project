from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import shutil
import uuid

app = FastAPI()

# 임시 폴더 경로
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)


# 파일 업로드 엔드포인트
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # 업로드된 파일이 HWP인지 확인
    if not file.filename.endswith('.hwp'):
        raise HTTPException(status_code=400, detail="HWP 파일만 업로드할 수 있습니다.")

    # 임시 파일 경로 생성
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(TEMP_DIR, unique_filename)

    # 파일 저장
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        print("파일이 저장되었습니다.")

    # 파일 삭제
    #os.remove(file_path)

    return {"filename": file.filename}
    #return {"filename": file.filename, "content": text}

