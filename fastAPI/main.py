from typing import Union

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import os,sys

# 폴더 리스트
folders = ['folder1', 'folder2', 'folder3', "./Preprocessing/separate"]

# 각 폴더를 sys.path에 추가
for folder in folders:
    folder_path = os.path.abspath(folder)
    sys.path.append(folder_path)

import separate_module_ver1 as sp 

app = FastAPI()

#get 할거
class ArticleAnalysis(BaseModel):
    article_number: int
    clause_number : int
    subclause_number: int                        
    Unfair: str
    Toxic: str
    explain: str

# 업로드 할 내용?
class SeparateResult(BaseModel):
    article_number: int
    # clause_number : int
    # subclause_number: int
    article_content: str
    
@app.post("/separate_result")
async def input_file(response_model: List[SeparateResult], file: UploadFile = File(...)):
    file: UploadFile = File(...)
    contents = await file.read()
    separate_text: sp.extract_and_modify_contract(contents)
    results = []
    for key, value in separate_text.items():
        result = SeparateResult(
            article_number=value.get('article_number'),
            # clause_number,  # 임의의 값, 실제 값에 맞게 설정
            # subclause_number,  # 임의의 값, 실제 값에 맞게 설정
            article_content=value.get('article_content')
        )
        results.append(result)
    
    # 모든 결과를 반환
    return results
        
        

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}


# @app.put("/items/{item_id}")
# def update_item(item_id: int, item: Item):
#     return {"item_name": item.name, "item_id": item_id}