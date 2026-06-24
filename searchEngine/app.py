import streamlit as st
#Libraries to perform search operation
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
#Libraries for wrapper around query search
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
#This is deprecated, we can use from langchain.agents import create_agent
from langchain_classic.agents import AgentType, initialize_agent
from langchain.agents import create_agent
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
from dotenv import load_dotenv
load_dotenv()

#nvidia_api_key = os.getenv("NVIDIA_API_KEY")

#Used the inbuild tool of wikipedia
api_wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wiki_wrapper)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchRun(name="Search")


## Create streamlit app
st.title("Enhanced Langchain Chat with search")

# sidebar setting
st.sidebar.title("Setting")
api_key = st.sidebar.text_input('Enter your NVIDIA API key :',type="password")
nvidia_api_key = api_key

# This is default message for search bot on first
if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hello! I am chatbot who can search from web. Let me know how can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatNVIDIA(nvidia_api_key=nvidia_api_key,model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")

    tools = [search,arxiv,wiki]

    search_agent = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        #response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        response=search_agent.run(prompt,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant','content':response})
        st.write(response)