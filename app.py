import streamlit as st
import os
from dotenv import load_dotenv

from utils.pandas_agent import ask_pandas_agent
from utils.sql_agent import ask_sql_agent

from utils.document_loader import load_documents
from utils.rag_pipeline import (
    create_vector_store,
    load_vector_store,
    create_qa_chain
)

from utils.csv_agent import analyze_operational_data

load_dotenv()

st.set_page_config(page_title="Enterprise Chat AI Agent")

st.title("⚡ Enterprise Chat AI Agent")

st.sidebar.header("Upload Enterprise Knowledge")

uploaded_files = st.sidebar.file_uploader(
    "Upload SOP PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("Build Knowledge Base"):

    os.makedirs("data/sops", exist_ok=True)

    for file in uploaded_files:

        with open(f"data/sops/{file.name}", "wb") as f:

            f.write(file.read())

    documents = load_documents("data/sops")

    create_vector_store(documents)

    st.sidebar.success("Knowledge Base Created Successfully!")

# Load Existing Vector DB

vectordb = load_vector_store()

qa_chain = create_qa_chain(vectordb)

st.header("Ask SOP / Operational Questions")

#query = st.text_input(
#    "Enter your question"
#)

## New Code for Pandas Agent and SQL Agent
st.write("Ask questions on SOP / Operational CSV datasets")

# Mode selection
mode = st.selectbox(
    "Choose Analysis Engine",
    ["SOP", "Pandas Agent", "SQL Agent"]
)

# User question
question = st.text_input(
    "Enter your question"
)

if st.button("Generate Answer"):

    if question:

        with st.spinner("Analyzing..."):

            try:

                if mode == "SOP":
                    answer = qa_chain.run(question)            
                elif mode == "Pandas Agent":
                    answer = ask_pandas_agent(question)
                else:
                    answer = ask_sql_agent(question)

                st.subheader("AI Response")
                st.success("Answer Generated")
                st.write(answer)

            except Exception as e:
                st.error(str(e))

#if st.button("Generate Answer"):

    #if query:

        #response = qa_chain.run(query)

        #try:
        #   response = qa_chain.run(query)
        #    print(response)

        #except Exception as e:
        #    print("Error:", e)

        #st.subheader("AI Response")

        #st.write(response)

# CSV Analytics Section

st.header("Operational Dataset Analysis")

csv_file = st.file_uploader(
    "Upload Operational CSV",
    type=["csv"]
)

if csv_file:

    os.makedirs("data/datasets", exist_ok=True)

    csv_path = f"data/datasets/{csv_file.name}"

    with open(csv_path, "wb") as f:

        f.write(csv_file.read())

    insights = analyze_operational_data(csv_path)

    st.subheader("Dataset Insights")

    st.json(insights)


# Building Operational CSV Knowledge Base
if st.button("Build Operational CSV Knowledge Base"):

    os.makedirs("data/datasets", exist_ok=True)

    for file in uploaded_files:

        with open(f"data/datasets/{file.name}", "wb") as f:

            f.write(file.read())

    documents_csv = load_documents("data/datasets")

    create_vector_store(documents_csv)

    st.success("Operational CSV Knowledge Base Created Successfully!")
