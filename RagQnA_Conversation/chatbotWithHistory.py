import os

from dotenv import load_dotenv

import streamlit as st

from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------- Setup --------------------

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

st.title("RAG Q&A Chatbot with PDF and Chat History")
st.text("Upload a PDF and ask questions with conversational memory.")


# -------------------- API key input --------------------

api_key = st.text_input("Enter your NVIDIA API key:", type="password")

if not api_key:
    st.warning("Enter your API key to continue.")
    st.stop()

# Use a stable, known chat model
llm = ChatNVIDIA(
    nvidia_api_key=api_key,
    model="nvidia/nemotron-4-70b-instruct",
    stream=True # Enable streaming
)

# -------------------- Session & history store --------------------

session_id = st.text_input("Session ID:", value="default_session")

if "store" not in st.session_state:
    st.session_state.store = {}


def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]


# -------------------- PDF upload & processing --------------------

file_upload = st.file_uploader(
    "Choose a PDF file",
    type="pdf",
    accept_multiple_files=False,
)

if not file_upload:
    st.info("Upload a PDF to start asking questions.")
    st.stop()

# Save uploaded file temporarily
temp_pdf = "./temp.pdf"
with open(temp_pdf, "wb") as f:
    f.write(file_upload.getvalue())
    file_name = file_upload.name

# Load and split PDF
loaded_docs = PyPDFLoader(temp_pdf).load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=200,
)
split_docs = text_splitter.split_documents(loaded_docs)

# Vector store & retriever
vector_store = Chroma.from_documents(split_docs, embedding=embeddings)
retriever = vector_store.as_retriever()


# -------------------- Prompts & chains --------------------

# History-aware question reformulation
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question, which might reference "
    "context from the chat history, formulate a standalone question that can "
    "be understood without the chat history. DO NOT answer the question. "
    "Reformulate the question if needed, otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt,
)

# Q&A prompt
system_prompt = (
    "You are a helpful assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say you don't know. Use a maximum of 3 sentences "
    "and keep the answer precise.\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

conversation_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# -------------------- Chat UI --------------------

user_input = st.text_input("Enter your question:")

if user_input:
    session_history = get_session_history(session_id)

    response = conversation_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )

    st.write("**Assistant:**", response["answer"])
    st.write("**Chat history (debug):**", session_history.messages)
    st.write("**Session store (debug):**", st.session_state.store)
