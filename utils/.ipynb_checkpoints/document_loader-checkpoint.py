import os
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.document_loaders import CSVLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(folder_path):

    documents = []

    for file in os.listdir(folder_path):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(folder_path, file)

            loader = PyPDFLoader(pdf_path)

            docs = loader.load()

            documents.extend(docs)

        elif file.endswith(".csv"):

            csv_path = os.path.join(folder_path, file)

            loader = CSVLoader(csv_path)

            docs = loader.load()

            documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(documents)

    return split_docs