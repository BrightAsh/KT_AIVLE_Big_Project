import pandas as pd

# 데이터 로드
data_MJ = pd.read_csv('./Data_Analysis/Data/unfair_sentence(MJ)+article.csv')
data_GS = pd.read_csv('./Data_Analysis/Data/unfair_sentence(GS)+article.csv')
data_GS_2 = pd.read_csv('./Data_Analysis/Data/unfair_sentence(GS)+article_2.csv')
#data_JS = pd.read_csv('./Data_Analysis/Data/toxic_sentence(JS).csv')
print(data_MJ.shape)
print(data_GS.shape)
print(data_GS_2.shape)
#print(data_JS.shape)
# 데이터 병합
merged_data = pd.concat([data_MJ, data_GS, data_GS_2], ignore_index=True)

# 중복된 행 개수 확인
num_duplicates = merged_data.duplicated(subset=['sentence']).sum()
print(f"중복된 행 개수: {num_duplicates}")

# 중복 제거
filtered_data = merged_data.drop_duplicates(subset=['sentence'], keep='first')

# 'label' 컬럼이 존재한다면 정수형으로 변환
filtered_data.loc[:, 'label'] = filtered_data['label'].astype(int)
filtered_data.loc[:, 'article'] = filtered_data['article'].astype(int)
print(len(filtered_data))
# 결과 저장
filtered_data.to_csv('./Data_Analysis/Data/unfair_sentence_merged.csv', index=False, encoding="utf-8-sig")



#log
# MJ
# MJ + GS
# MJ + GS + GS_2
