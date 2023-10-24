import streamlit as st
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from dotenv import load_dotenv
import os
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.llms import HuggingFaceHub


DB_FAISS_PATH = '/home/ubuntu/ProBot/Data/vectorstore/db_faiss'

custom_prompt_template = """
You are a helpful assistant, you will use the provided context to answer the questions.
Read the given context before answering questions and think step by step.
If you can not answer a user question based the provided context just say Sorry I don't know that but I'm still learning.
Do not use and other information for answering user. Provide a detailed answer to the user.
Context: {context}
Question: {question}
Only return the relevant answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                                      chain_type='stuff',
                                                      retriever=db.as_retriever(search_kwargs={'k': 2}),
                                                      return_source_documents=True,
                                                      chain_type_kwargs={'prompt': prompt}
                                                    )

    return qa_chain


#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = HuggingFaceHub(
        repo_id = "HuggingFaceH4/zephyr-7b-alpha",
        model_kwargs={
           "temperature": 0.6,
           "max_length": 512,
           "max_new_tokens" : 800
        }
    )
    return llm


#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-distilbert-cos-v1",
                                    model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response["result"]


def main():
  load_dotenv()
  st.sidebar.title("ProBot 🤖")
  st.sidebar.write("Probot is a Generative-AI bot which can reslove your queries related to the microservices which are present in our project.")
  st.sidebar.write("⚫ Check the official docs page:-")
  st.sidebar.write("⚫ Doc:-")
  st.title("ProBot 🤖")
  st.subheader("A Generative-AI Bot.")
  text_inputs=st.text_input("Ask your query....")
  if st.button("Ask query"):
    with st.spinner("thinking..."):
      if len(text_inputs)>0:
        answer=final_result(text_inputs)
        st.success(answer)

if __name__=="__main__":
  main()
