import streamlit as st
from pathlib import Path
from langchain_classic.agents import create_sql_agent
from langchain_classic.sql_database import SQLDatabase
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_classic.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# UI field creation
st.set_page_config(page_title="Lanchain: Chatbot with SQL DB ", page_icon="🦈")
st.title("🦈 Lanchain: Chatbot with SQL DB  ")

injection_warning = ""

LOCAL_DB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"
db_uri=None



radio_optn = ["Local SqlLite3 DB","MySQL server DB"]
selected_db_optn = st.sidebar.radio("Select DB source ",options=radio_optn)

@st.cache_resource(ttl="2h")
def configure_db(db_uri,mysql_host=None,mysql_user=None,mysql_pwd=None,mysql_db=None):
    if db_uri==LOCAL_DB:
        db_filePath = (Path(__file__).parent/"student.db").absolute()
        print(f"Db Path : {db_filePath}")
        connector = lambda:sqlite3.connect(f"file:{db_filePath}?mode=ro",uri=True)
        return SQLDatabase(create_engine("sqlite:///",creator=connector))
    elif db_uri==MYSQL:
        if not (mysql_host and mysql_user and mysql_pwd and mysql_db):
            st.warning("Enter the My Sql Db info")
            st.stop()
            st.success("Thankyou for entering all Db details")
            return SQLDatabase(create_agent(f"mysql+mysqlconnector://{mysql_user}:{mysql_pwd}@{mysql_host}/{mysql_db}"))


if radio_optn.index(selected_db_optn)==1:
    db_uri = MYSQL
    mySqlHost=st.sidebar.text_input("Enter MySQL host")
    mySqlUser=st.sidebar.text_input("Enter MySql User")
    mySqlPwd =st.sidebar.text_input("Enter MySql password",type="password")
    mysqlDb=st.sidebar.text_input("Enter MySql DB")
    
else:
    db_uri=LOCAL_DB



if db_uri==MYSQL:
        db = configure_db(db_uri,mySqlHost,mySqlUser,mySqlPwd,mysqlDb)
else:
    db=configure_db(db_uri)

api_key = st.sidebar.text_input("Enter NVIDIA key", type="password")

if not api_key:
    st.info("Enter the valid DB information and NVIDIA key")
    st.stop()

# LLM model
llm = ChatNVIDIA(
        nvidia_api_key=api_key,
        model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
        #model="nvidia/nemotron-3-ultra-550b-a55b"
    )


# toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_executor_kwargs={
        "handle_parsing_errors": True,
    }
)

if "messages" not in st.session_state or st.sidebar.button("clear chat history"):
    st.session_state["messages"]=[{"role":"assistant","content":"How can I help you?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):st.write(msg["content"])

user_query=st.chat_input(placeholder="Enter user query")

if user_query:
    st.session_state.messages.append({"role":"user","content":user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback=StreamlitCallbackHandler(st.container())
        try:
            response = agent.run(user_query,callbacks=[streamlit_callback])
            st.session_state.messages.append({"role":"assistant","content":response})
            st.write(response)
        except Exception as e:
            error_msg = f"Error: {str(e)}\n\nPlease try asking your question in a different way. Available table: STUDENT (NAME, CLASS, SECTION, MARKS)"
            st.error(error_msg)
            st.session_state.messages.append({"role":"assistant","content":error_msg})