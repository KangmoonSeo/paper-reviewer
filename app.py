import streamlit as st
from llama_index import SimpleDirectoryReader
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex
from dotenv import load_dotenv
import os
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

system_prompt = '''You are the paper reviewer, and your job is to answer questions about the attached paper.
Based on the content of the paper, please answer future questions.'''

def main():
    st.header("Paper review Application")
    uploaded_file = st.file_uploader("Upload your file here...")
    
    uploaded_file_path = None  
    
    if uploaded_file:
        file_name = os.path.join('./data', uploaded_file.name)
        uploaded_file_path = file_name 
        with open(file_name, "wb") as f:
            f.write(uploaded_file.read())
            
        if st.button("Delete Uploaded File"):
            if os.path.exists(uploaded_file_path):
                os.remove(uploaded_file_path)
                st.success("File has been deleted.")
                uploaded_file_path = None 
    
    if uploaded_file_path:
        reader = SimpleDirectoryReader(input_dir='./data', recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt=system_prompt))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        
        query = st.text_input("Ask questions related to the paper.")
        if query:
            chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
            response = chat_engine.chat(query)
            st.write(response.response)

if __name__ == '__main__':
    main()
