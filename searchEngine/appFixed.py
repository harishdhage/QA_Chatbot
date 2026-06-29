import streamlit as st
# Libraries to perform search operation
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
# Libraries for wrapper around query search
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
# Note: initialize_agent is legacy but kept here to match your setup
from langchain_classic.agents import AgentType, initialize_agent
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_nvidia_ai_endpoints import ChatNVIDIA
#import os
#from dotenv import load_dotenv

#load_dotenv()

# Used the inbuilt tool of wikipedia
api_wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wiki_wrapper)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchRun(name="Search")

## Create streamlit app
st.title("Enhanced Langchain Chat with search")

# sidebar setting
st.sidebar.title("Setting")
# 1. Fallback to .env if user hasn't typed anything in the sidebar yet
user_api_key = st.sidebar.text_input('Enter your NVIDIA API key :', type="password")
nvidia_api_key = user_api_key #if user_api_key else os.getenv("NVIDIA_API_KEY")

# This is default message for search bot on first load
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I am a chatbot who can search the web. How can I help you today?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 2. Guard rail to ensure API key exists before invoking the model
    if not nvidia_api_key:
        st.error("Please add your NVIDIA API key in the sidebar or .env file to proceed.")
        st.stop()

    try:
        # Initializing LLM with the active API key
        llm = ChatNVIDIA(
            nvidia_api_key=nvidia_api_key, 
            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
        )

        tools = [search, arxiv, wiki]

        search_agent = initialize_agent(
            tools, 
            llm, 
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
            handle_parsing_errors=True # Fixed typo: handling_parsing_errors -> handle_parsing_errors
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            
            # 3. FIX: Pass the single 'prompt' string instead of the entire messages list
            response = search_agent.run(prompt, callbacks=[st_cb])
            
            st.session_state.messages.append({'role': 'assistant', 'content': response})
            st.write(response)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")