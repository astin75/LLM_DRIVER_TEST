import os 
import openai 
import tiktoken


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma 
from langchain.document_loaders import PyPDFLoader

from langchain.embeddings import HuggingFaceEmbeddings 
from langchain_core.embeddings import Embeddings

from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler 


os.environ["OPENAI_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxx"

def tiktoken_len(text: str) -> int:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)  

def load_pdf(filename_list: list) -> list:
    documents = []
    for filename in filename_list:
        loader = PyPDFLoader(filename)
        pages = loader.load()
        documents.extend(pages)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512,
                                                    length_function=tiktoken_len)#min_tokens=100, max_tokens=200)
    texts = text_splitter.split_documents(documents)
    return texts

def load_huggingface_embeddings(device: str = "cpu", model_name: str = "jhgan/ko-sbert-nli", ) ->Embeddings:
    #https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.huggingface.HuggingFaceEmbeddings.html?highlight=hugging
    #https://m.blog.naver.com/hist0134/220977173543
    #https://github.com/jhgan00/ko-sentence-transformers
    #https://huggingface.co/jhgan/ko-sbert-nli
    model_kwargs = {"device": device}
    encode_kwargs ={'normalize_embeddings': True}
    hf = HuggingFaceEmbeddings(model_name=model_name,
                                model_kwargs=model_kwargs,
                                encode_kwargs=encode_kwargs)

    return hf

def make_vector_store(docs: list, llm_embedding: Embeddings, persist_directory: str = None) -> Chroma:
    #https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.chroma.Chroma.html?highlight=chroma#langchain_community.vectorstores.chroma.Chroma.from_documents
    vector_store = Chroma.from_documents(documents=docs, embedding=llm_embedding, persist_directory=persist_directory)
    if persist_directory is not None:
        vector_store.persist()
        vector_store = None 
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=llm_embedding)
    return vector_store

def load_vector_store(persist_directory: str, llm_embedding: Embeddings) -> Chroma:
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=llm_embedding)
    return vector_store

def load_chat_model(model_name: str = "gpt-3.5-turbo") -> ChatOpenAI:
    #StreamingstdoutCallback: If you only want the final output of an agent to be streamed
    opeanai = ChatOpenAI(model_name=model_name,
                         streaming=True, callbacks=[FinalStreamingStdOutCallbackHandler()],
                         temperature= 0)
    return opeanai