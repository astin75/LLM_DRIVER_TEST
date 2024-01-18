import json
import pandas as pd



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
    
 
        
        

tabs = pd.ExcelFile("asset/240118_d-test데이터셋_v1.xlsx").sheet_names 
df = pd.read_excel("asset/240118_d-test데이터셋_v1.xlsx",
                   sheet_name=tabs[1])
del tabs
with open("exam/asset/240119_test_answer_exam.json") as f:
    answer_json = json.load(f)

binary_score, weight_score = score(answer_json, df, isopenbook=False)
print(f"binary_score: {binary_score}", f"weight_score: {weight_score}")