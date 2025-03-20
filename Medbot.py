import streamlit as st

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

DB_FAISS_PATH = "vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
        embedding_model = HuggingFaceEmbeddings(model_name ='sentence-transformers/all-MiniLM-L6-v2') 
        db = FAISS.load_local(folder_path=DB_FAISS_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)
        return db

def set_custom_prompt(custom_prompt_template):
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", custom_prompt_template),
        ("human", "{input}")
    ]
)
    return prompt

def load_llm(modelID, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id = modelID,
        temperature=0.3,
        model_kwargs = {"token":HF_TOKEN,
                        "max_length": 512}
        
    )
    return llm


def main():
    st.title("MedBot")
    
    # Create a session state to store messages
    if 'messages' not in st.session_state:
        st.session_state.messages =[]
    
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Take user prompt
    query = st.chat_input("Enter query:")

    if query:
        st.chat_message('user').markdown(query)
        st.session_state.messages.append({'role':'user', 'content':query})

        custom_prompt_template =  """
                Use the information in the given context to answer the question acting as a doctor. Talk to the user directly.
                Stick to the context and answer the question. 
                Do not make small talk. If the answer is not known, say "I don't know".
                Context: {context}
                Please provide the answer directly.
                """
        modelID = "mistralai/Mistral-7B-Instruct-v0.3"

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load vector store")

            retriever = vectorstore.as_retriever(search_kwargs={"k":3})
            llm = load_llm(modelID, HF_TOKEN)


            prompt = set_custom_prompt(custom_prompt_template)

            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            chain = create_retrieval_chain(retriever, question_answer_chain)

            # Query the chain
            reponse = chain.invoke({"input": query})
            result ="Certainly! "+ reponse['answer']

            #response = "Hi! I am MedBot"
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role':'assistant', 'content':result})

        except Exception as e:
            st.error(f"An error occurred: {e}")
        


if __name__ == "__main__":
    main()