import os 
import json
import time
from json_numpy import default
import json_numpy

from core.utils import load_pdf, load_huggingface_embeddings, make_vector_store, load_vector_store, load_chat_model
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langchain.embeddings import OpenAIEmbeddings

pdf_path1 = "asset/c_도로교통법(법률)(제19745호)(20231024).pdf"
#pdf_path1 = "asset/b_도로교통법규 정리자료.pdf"
pdf_path2 = "asset/d_도로교통법 시행규칙(행정안전부령)(제00431호)(20231117).pdf"
pdf_path3 = "asset/b_도로교통법규 정리자료.pdf"
version = "e"
texts = load_pdf([pdf_path2, pdf_path1])


#hf_embedding = load_huggingface_embeddings(device='cuda:2')
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",)
start = time.time()
vector_db_path = "./asset/driver_license_text"

if os.path.exists(vector_db_path):
    vector_store = load_vector_store(vector_db_path, embeddings)
else:
    vector_store = make_vector_store(texts, embeddings, persist_directory=f"./asset/{version}_driver_license_text")
print(f"make_vector_store: {time.time() - start}")

chat_model = load_chat_model(model_name="gpt-4-1106-preview")


'''
search_type : mmr
⚪️ MMR (Maximal Marginal Relevance)

MMR이란 검색 엔진 내에서 본문 검색 관련하여 검색에 따른 결과의 다양성과 연관성을 조절하는 방법이다.

텍스트 요약 작업에서 중복성을 최소화하고 결과의 다양성을 극대화하기 위한 옵션 !
문서와 가장 유사한 키워드를 선택 → 문서와 비슷하면서도 이미 선택한 키워드와 비슷하지 않은 새 후보를 반복적으로 선택

k: Number of Documents to return. Defaults to 4.
fetch_k: Number of Documents to fetch to pass to MMR algorithm
'''

# Build prompt
template = """ 
{context}
Question: {question} 
Helpful Answer:
- If you're not sure, "I don't know the answer to that question", thank you for your service .
- answer the following question with number of answer and reason from document.
"""
#QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain

qa = RetrievalQA.from_chain_type(llm = chat_model,
                 chain_type = "stuff",
                #chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT},
                 retriever = vector_store.as_retriever(
                     search_type = "mmr",
                     search_kwargs = {'k':3, 'fetch_k': 10}),
                 return_source_documents = True,)

Helpful_Answer = "정답을 버!"
Helpful_Answer = """
- 정답을 -> [] 안에 번호만 넣어줘 
"""
question = f"""
다음 중 총중량 1.5톤 피견인 승용자동차를 4.5톤 화물자동차로 견인하는 경우
필요한 운전면허로 바람직하지 않은 것은?
① 제1종 대형면허 및 소형견인차면허
② 제1종 보통면허 및 대형견인차면허
③ 제1종 보통면허 및 소형견인차면허
④ 제2종 보통면허 및 대형견인차면허

{Helpful_Answer}
"""
# question = f"""
# 2. 도로교통법령상 운전면허증 발급에 대한 설명으로 옳지 않은 것은?
# ① 운전면허시험 합격일로부터 30일 이내에 운전면허증을 발급받아야 한다.
# ② 영문운전면허증을 발급받을 수 없다.
# ③ 모바일운전면허증을 발급받을 수 있다.
# ④ 운전면허증을 잃어버린 경우에는 재발급 받을 수 있다. 

# {Helpful_Answer}
# """
qa.save("asset/yaml/b_driver_license.yaml")

with open("/home/sungeun/nearth/lang_chain/asset/test_docs_p.json", "r") as f:
    test_docs = json.load(f)
    
output_dict = {}
break_number = 1
start = time.time()
count = 0
for k, v in test_docs.items():
    print(k)
    question = '\n'.join(v["question"])
    intput_qustion =f"""
    {question}
    {Helpful_Answer}
    """
    result = qa({"query": intput_qustion})
    temp_dict = {}
    for idx, doc in enumerate(result['source_documents']):
        temp_dict[idx] = doc.to_json()
    output_dict[k] = {
        "question": intput_qustion,
        "answer_number": v["answer_number"],
        "solution": v["solution"],
        "predict" : result['result'],
        "source_documents": temp_dict
    }
    if int(k) > 99:
        "break"
        break
    count += 1
output_dict["time"] = round(time.time() - start, 2)
with open(f"/home/sungeun/nearth/lang_chain/asset/predict_result_v_1_{version}_v4.json", "w") as f:
    json.dump(output_dict, f, ensure_ascii=False, indent=4)