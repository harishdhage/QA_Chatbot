import os
import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import create_agent
from langchain_classic.chains import LLMMathChain
from langchain_classic.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents import Tool,initialize_agent
from dotenv import load_dotenv
from langchain_classic.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="LangChain: Math problem solver GPT",
    page_icon="🦈",
)

st.title("🦈 LangChain: Math problem solver GPT")
#st.subheader("Paste a URL or YouTube link below to get a summary")


# Sidebar – API key


api_key = os.getenv("NVIDIA_API_KEY")
llm = ChatNVIDIA(
        model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        nvidia_api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=0.3,
    )
#Initialize tools
wiki_wraper = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="Wikipedia",
    func=wiki_wraper.run,
    description="A tool to search in web to solve math problem and varous alternative solutions"
)

# Initialize the math tool
math_chain = LLMMathChain.from_llm(llm=llm)
calc= Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool to answer the math related question. Inly mathematical expressions."
)

prompt="""
You are agent to solve the taks related to Mathematical questions. Also provide the detailed logical solution.
Question:{question}
Answer:
"""

promt_template = PromptTemplate(
    input_variables=['question'],
    template=prompt
)

## Math problem tool
chain = llm | promt_template

reasoning_tool = Tool(
    name="Reasoning",
    func=chain.run,
    description="Tool for Q & A logocal reasoning"
)

# initialize agents
assistant_agent = initialize_agent(
    tools=[wiki_tool,calc,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_error=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hello, I am Math solver Chatbot. I can provide the solution for your mathematical problems"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

## function to handle generate response
'''def generate_response(user_question):
    response = assistant_agent.invoke({"input":user_question})
    return response'''

#Lets start interaction
question = st.text_area("Enter your questions")
if st.button("Get answer"):
    if question:
        with st.spinner("In progress..."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.invoke(st.session_state.messages, callbacks=[st_cb])

            st.session_state.messages.append({"role":"assistant","content":response})
            st.write('### Response')
            st.success(response)
    else:
        st.warning("Enter the input")