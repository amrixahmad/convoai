import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css,bot_template,user_template

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    
    return text

def get_text_chunks(raw_text):
    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks=text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(text_chunks):
    embeddings=OpenAIEmbeddings()
    vector_store=FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    llm=ChatOpenAI()
    memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handleUserInput(user_question):
    response=st.session_state.conversation({'question':user_question})
    # st.write(response)
    st.session_state.chat_history=response["chat_history"]
    # st.write(st.session_state.chat_history)

    for i, message in enumerate(st.session_state.chat_history):
        
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)

def main():
    load_dotenv()    
    st.set_page_config(page_title="Chat with multiple pdfs",page_icon=":books:")    
    st.write(css,unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat with multiple pdfs :books:")
    user_question=st.text_input("Ask a question about your documents:")

    if user_question:
        handleUserInput(user_question)    
    
    with st.sidebar:
        st.subheader("Your documents:")
        pdf_docs=st.file_uploader(
            "Upload your pdf here and click process",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text=get_pdf_text(pdf_docs)                
                text_chunks=get_text_chunks(raw_text)
                # st.write(text_chunks)
                vector_store=get_vector_store(text_chunks)
                # st.write(type(vector_store))
                st.session_state.conversation=get_conversation_chain(vector_store)
    
    # print(OPENAI_API_KEY)


if __name__=="__main__":
    main()