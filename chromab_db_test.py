from core.utils import tiktoken_len, load_vector_store
from langchain.embeddings import OpenAIEmbeddings
version = "h"
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",)
vector_store = load_vector_store(f"./asset/{version}_driver_license_text", embeddings)

query = "What did the president say about Ketanji Brown Jackson"
query = "도로교통법 제89조: 제89조"
query = " 경찰청장에게 통보하여야 하는 개인정보의 내용 및 통보방법과 그 밖에 개인정보의 통보에 필요한\n사항은 대통령령으로 정한다."
#docs = vector_store.similarity_search(query)
docs = vector_store.similarity_search_with_relevance_scores(query,k=3,fetch_k=3)
#docs = vector_store.similarity_search_with_score(query,k=3,fetch_k=3)
# print results
#print(docs[0].page_content)
print(docs)