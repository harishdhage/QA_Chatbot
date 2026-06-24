import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os 
from dotenv import load_dotenv
load_dotenv()

#Lanchsmith tracking
os.environ["LANGCHAIN-API-KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot with Nvidia"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are helpfull assistance, you answer the question asked by user"),
        ("user","Questions :{question}")
    ]
)

def generate_response(question, api_key, llm, temperature, max_tokens):
    api_key = api_key
    llm=ChatNVIDIA(model=llm)
    output_parser = StrOutputParser()
    chain= prompt|llm|output_parser
    answer = chain.invoke({"question":{question}})
    return answer

## Create streamlit app
st.title("Enhanced Nvidia Q&A Chatbot")

st.sidebar.title("Setting")
api_key = st.sidebar.text_input('Enter your API key :',type="password")
llm = st.sidebar.selectbox("Select an Nvdia AI model : ",["nvidia/nemotron-3-nano-omni-30b-a3b-reasoning","minimaxai/minimax-m2.7","stepfun-ai/step-3.5-flash"])
temperature = st.sidebar.slider("Temperature ",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Tokens",min_value=50,max_value=400,value=150)

#Main interface
st.write("Ask you question")
user_input = st.text_input("You : ")

if user_input:
    response = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write(response)
elif user_input:
    st.write("Please enter your question..")