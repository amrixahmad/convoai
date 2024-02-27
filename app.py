import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

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

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple pdfs",page_icon=":books:")
    st.header("Chat with multiple pdfs :books:")
    st.text_input("Ask a question about your documents:")
    
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
    
    # print(OPENAI_API_KEY)


if __name__=="__main__":
    main()