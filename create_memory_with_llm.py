# Setup Mistral LLM using HugginFace

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# HuggingFace Repository ID for the model
modelID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(modelID):
    llm = HuggingFaceEndpoint(
        repo_id = modelID,
        temperature=0.3,
        model_kwargs = {"token":HF_TOKEN,
                        "max_length": 512}
        
    )
    return llm

# Connect LLM with FAISS and create chain

FAISS_PATH = "vectorstore/db_faiss"

custom_prompt_template =  """
Use the information in the given context to answer the question acting as a doctor. Talk to the user directly.
Stick to the context and answer the question. 
Do not make small talk. If the answer is not known, say "I don't know".
Context: {context}
Please provide the answer directly.
"""

def set_custom_prompt(cutom_prompt_template):
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", custom_prompt_template),
        ("human", "{input}")
    ]
)
    return prompt

embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(folder_path= FAISS_PATH, embeddings= embedding_model, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_kwargs={"k":3})
llm = load_llm(modelID)

prompt = set_custom_prompt(custom_prompt_template)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

# Query the chain
query = input("Enter your query: ")
reponse = chain.invoke({"input": query})
print("Result: ", reponse['answer'])
