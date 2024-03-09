import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from PyPDF2 import PdfReader
import docx
from htmlTemplates import bot_template, user_template, css
import os
import const

def get_text(file_upload):
    if file_upload.type == 'application/pdf':
        try:
            pdf_reader = PdfReader(file_upload)
            # Initialize an empty string to store the text
            text = ""
            # Iterate over each page of the PDF and extract text
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except AttributeError:
            st.error("Please upload a PDF file.")
    elif file_upload.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        try:
            doc = docx.Document(file_upload)
            text = ""
            # Iterate over each paragraph of the Word document and extract text
            for para in doc.paragraphs:
                text += para.text + '\n'
            return text
        except AttributeError:
            st.error("Please upload a Word file.")
    elif file_upload.type == 'text/plain':
        try:
            text = file_upload.getvalue().decode("utf-8")
            return text
        except AttributeError:
            st.error("Please upload a text file.")

def get_chunk_text(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    os.environ["OPENAI_API_KEY"] = const.API_KEY
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(question):
    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with Your own PDFs', page_icon=':books:')
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    st.header('Chat with Your own PDFs (Test) :books:')
    
    file_path = st.file_uploader("Upload your file", type=['pdf', 'docx', 'txt'])

    if file_path is not None:
        question = st.text_input("Ask anything based on your text: ")
        if question:
            handle_user_input(question)
        with st.spinner("Processing your text..."):
            raw_text = get_text(file_path)
            if raw_text:
                text_chunks = get_chunk_text(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.write("DONE")
                st.session_state.conversation = get_conversation_chain(vector_store)

if __name__ == '__main__':
    main()
