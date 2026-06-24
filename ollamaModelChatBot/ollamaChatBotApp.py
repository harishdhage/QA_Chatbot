from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

#Lanchsmith tracking
os.environ["LANGCHAIN-API-KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot with Ollama"

prompt = ChatPromptTemplate.from_messages([
    ("system","You are helpfull assistance, you answer the question asked by user"),
    ("user","Questions :{question}")
])

def generate_response(question, engine, temperature, max_tokens):
    llm=Ollama(model=engine)
    output_parser = StrOutputParser()
    chain= prompt|llm|output_parser
    answer = chain.invoke({"question":{question}})
    return answer

## Create streamlit app
st.title("Enhanced Ollama Q&A Chatbot")

st.sidebar.title("Setting")
engine = st.sidebar.selectbox("Select an Nvdia AI model : ",["qwen3.5:0.8b","qwen3.5"])
temperature = st.sidebar.slider("Temperature ",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Tokens",min_value=50,max_value=400,value=150)

#Main interface
st.write("Ask you question")
user_input = st.text_input("You : ")

if user_input:
    response = generate_response(user_input, engine, temperature, max_tokens)
    st.write(response)
elif user_input:
    st.write("Please enter your question..")
