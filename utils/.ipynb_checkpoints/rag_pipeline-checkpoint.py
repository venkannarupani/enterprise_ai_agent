from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
#from langchain.chains import RetrievalQA
from langchain_classic.chains import RetrievalQA
#from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_community.llms import Ollama

VECTOR_DB_PATH = "vectorstore"

def create_vector_store(documents):

    embeddings = OpenAIEmbeddings()
    
    #embeddings = HuggingFaceEmbeddings(
    #model_name="sentence-transformers/all-MiniLM-L6-v2"
    #)

    #embeddings = HuggingFaceEmbeddings(
    #model_name="sentence-transformers/all-MiniLM-L6-v2",
    #model_kwargs={'device': 'cpu'}
    #)
    
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    vectordb.persist()

    return vectordb


def load_vector_store():

    embeddings = OpenAIEmbeddings()
    
    #embeddings = HuggingFaceEmbeddings(
    #model_name="sentence-transformers/all-MiniLM-L6-v2"
    #)

    #embeddings = HuggingFaceEmbeddings(
    #model_name="sentence-transformers/all-MiniLM-L6-v2",
    #model_kwargs={'device': 'cpu'}
    #)
    
    vectordb = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )

    return vectordb


def create_qa_chain(vectordb):

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini"
    )

    #llm = Ollama(model="llama3")
    #llm = Ollama(
    #model="llama3",
    #temperature=0,
    #num_ctx=2048,
    #model="qwen2.5:3b",        
    #base_url="http://localhost:11434"
    #)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    return qa_chain