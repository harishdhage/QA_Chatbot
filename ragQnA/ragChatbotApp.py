import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import streamlit as st

load_dotenv()
# os.environ["NVIDIA_API_KEY"]=os.environ("NVIDIA_API_KEY")
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
#Ollama runs locally, hence it take lot of time to process
OLLAMA_EMBEDDINGS_MODEL = os.getenv("OLLAMA_EMBEDDINGS_MODEL", "nomic-embed-text-v2-moe")


llm = ChatNVIDIA(nvidia_api_key=nvidia_api_key,model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided the context only.
    Provide the most accurate response based on the question asked 
    <context>
    {context}
    <context>
    Question:{input} 
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDINGS_MODEL)
        loader_path = Path(__file__).resolve().parent / "research_papers"
        st.session_state.loader = PyPDFDirectoryLoader(loader_path)
        st.session_state.documents = st.session_state.loader.load()

        if not st.session_state.documents:
            st.error(f"No PDF documents found in {loader_path}")
            return

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_document = st.session_state.text_splitter.split_documents(st.session_state.documents[:50])

        if not st.session_state.final_document:
            st.error("No text chunks were created from the loaded documents.")
            return

        try:
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_document, st.session_state.embeddings)
        except Exception as e:
            st.error(f"Failed to create vector store: {e}")
            return

st.title('RAG Chatbot')
user_prompt = st.text_input("Enter your question")

if st.button('Document Embedding'):
    create_vector_embedding()
    st.write("Vector DB is ready")

import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)

    if "vectors" not in st.session_state:
        st.error("Please build document embeddings first by clicking Document Embedding.")
    else:
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start_time = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        print(f"Response time : {time.process_time()-start_time}")
        st.write(response['answer'])

    with st.expander("Document similarity search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('----------------------')