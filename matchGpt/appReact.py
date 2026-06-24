import os
import streamlit as st
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_classic.agents import initialize_agent, Tool
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.prompts import PromptTemplate


# ---------------------------
# Setup
# ---------------------------

load_dotenv()

st.set_page_config(
    page_title="Math Solver Agent",
    page_icon="🧮",
)

st.title("🧮 Math Solver Agent (Classic LangChain)")
st.caption("Ask any math question. I can calculate, reason, and search Wikipedia.")


# ---------------------------
# NVIDIA LLM
# ---------------------------

api_key = st.sidebar.text_input(
    "NVIDIA API Key",
    value=os.getenv("NVIDIA_API_KEY", ""),
    type="password",
)

if not api_key:
    st.warning("Enter your NVIDIA API key to continue.")
    st.stop()

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    nvidia_api_key=api_key,
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=0.2,
)


# ---------------------------
# Tools
# ---------------------------

# Wikipedia tool
wiki = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="Wikipedia",
    func=wiki.run,
    description="Search Wikipedia for background information.",
)

# Calculator tool (Python eval)
def python_calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

calc_tool = Tool(
    name="Calculator",
    func=python_calculator,
    description="Evaluate mathematical expressions using Python.",
)

# Reasoning tool (LLM)
reasoning_prompt = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are a careful math tutor. Show step-by-step reasoning.\n\n"
        "Question: {question}\n\nAnswer:"
    ),
)

def reasoning_func(question: str) -> str:
    prompt = reasoning_prompt.format(question=question)
    resp = llm.invoke(prompt)
    return resp.content

reasoning_tool = Tool(
    name="Reasoning",
    func=reasoning_func,
    description="Use this for step-by-step logical reasoning.",
)

tools = [wiki_tool, calc_tool, reasoning_tool]


# ---------------------------
# Classic Agent (your version supports this)
# ---------------------------

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,
)


# ---------------------------
# Chat UI
# ---------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I can solve math problems step-by-step."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# ---------------------------
# User Input
# ---------------------------

user_question = st.chat_input("Enter your math question...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    st.chat_message("user").write(user_question)

    with st.spinner("Thinking..."):
        result = agent.run(user_question)

    st.session_state.messages.append({"role": "assistant", "content": result})
    st.chat_message("assistant").write(result)
