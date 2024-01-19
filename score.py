import os
import json
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import dotenv_values
env = dotenv_values(".env")
os.environ["OPENAI_API_KEY"] = env["OPENAI_API_KEY"]

def cosine_similarity(sentence1, sentence2):
    embed_1 = openai_embeddings(sentence1)
    embed_2 = openai_embeddings(sentence2)
    similarity = np.dot(embed_1, embed_2) / (np.linalg.norm(embed_1) * np.linalg.norm(embed_2))
    return similarity
    

def openai_embeddings(input: str, model_name: str ="text-embedding-ada-002"):
    client = OpenAI()
    #https://platform.openai.com/docs/guides/embeddings/use-cases
    response = client.embeddings.create(
        model = model_name,
        input=[input])
    return response.data[0].embedding

def score(answer_json, gt_df, isopenbook=False):
    binary_score_list = []
    weight_score_list = []
    total_weight_score = 0
    for question_number in answer_json.keys():
        temp_df = gt_df.loc[gt_df['문제번호']==int(question_number)]       
        if isopenbook:
            if temp_df.iloc[0]["구분"] == "없음":
                continue
        question = answer_json[question_number]["question"]
        gt_number = answer_json[question_number]["gt_number"]
        predict_number = answer_json[question_number]["predict_number"]
        predict_solution = answer_json[question_number]["predict_solution"]
        rag_confidence = answer_json[question_number]["rag_confidence"]
        rag_content = answer_json[question_number]["rag_content"]    
        
             
        thinking = temp_df.iloc[0]["사고력"]
        memorization = temp_df.iloc[0]["암기력"]
        gt_weight_score = int(thinking) + (memorization)
        total_weight_score+=gt_weight_score
                
        if set(gt_number) == set(predict_number):
            binary_score_list.append(1)
            weight_score_list.append(gt_weight_score)
        else:
            binary_score_list.append(0) 
            weight_score_list.append(0)
    binary_score = sum(binary_score_list) / len(binary_score_list)
    weight_score = sum(weight_score_list) / total_weight_score
    return binary_score, weight_score

test1 = """
1. 다음 중 총중량 1.5톤 피견인 승용자동차를 4.5톤 화물자동차로 견인하는 경우 필요한 운전면허에 해당하지 않은 것은?
1 제1종 대형면허 및 소형견인차면허 2 제1종 보통면허 및 대형견인차면허 3 제1종 보통면허 및 소형견인차면허 4 제2종 보통면허 및 대형견인차면허
"""
test1 = """
5. 도로교통법상 연습운전면허의 유효 기간은?
1 받은 날부터 6개월 2 받은 날부터 1년 3 받은 날부터 2년 4 받은 날부터 3년
"""
test2 =  """
■ 해설:도로교통법 시행규칙 별표18 총중량 750킬로그램을 초과하는 3톤 이하의 피견인 자동차를 견인하기 위해서는 견인
하는 자동차를 운전할 수 있는 면허와 소형견인차면허 또는 대형견인차면허를 가지고 있어야 한다.
"""
similarity = cosine_similarity(test1, test2)
print(similarity)

         
# tabs = pd.ExcelFile("asset/240118_d-test데이터셋_v1.xlsx").sheet_names 
# df = pd.read_excel("asset/240118_d-test데이터셋_v1.xlsx",
#                    sheet_name=tabs[1])
# del tabs
# with open("exam/asset/240119_test_answer_exam.json") as f:
#     answer_json = json.load(f)

# binary_score, weight_score = score(answer_json, df, isopenbook=False)
# print(f"binary_score: {binary_score}", f"weight_score: {weight_score}")