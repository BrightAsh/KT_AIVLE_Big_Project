# 리소스 실행 코드
import nltk
import os

# 'punkt' 리소스가 저장된 경로
nltk_data_path = os.path.abspath("/usr/lib/nltk_data")
punkt_tab_data_path = os.path.abspath("/usr/lib/nltk_data")
# NLTK 리소스 경로에 추가
nltk.data.path.append(nltk_data_path)
nltk.data.path.append(punkt_tab_data_path)
# 'punkt' 리소스가 사용 가능한지 확인하고 없으면 다운로드
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)
try:
    nltk.data.find('punkt_tab')
except LookupError:
    nltk.download('punkt_tab', download_dir=punkt_tab_data_path)