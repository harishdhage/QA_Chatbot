import os
import streamlit as st
from urllib.parse import urlparse

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Updated import
from langchain_huggingface import HuggingFaceEndpoint


# ---------------------------
# Helpers
# ---------------------------

def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(
    page_title="LangChain: Summarize any URL or YouTube video",
    page_icon="🦈",
)

st.title("🦈 LangChain: Summarize any URL or YouTube video")
st.subheader("Paste a URL or YouTube link below to get a summary")


# Sidebar – API key
with st.sidebar:
    hf_api_key = st.text_input(
        "Enter your Huggingface API key",
        value="",
        type="password",
    )
    st.markdown(
        "Model: `microsoft/FastContext-1.0-4B-SFT\n\n"
        "Make sure your key has access to this model."
    )


# Main input
generic_url = st.text_input(
    "Enter the URL to be summarized",
    value="",
    label_visibility="visible",
    placeholder="https://example.com or https://www.youtube.com/watch?v=...",
)


# ---------------------------
# LLM + Modern LCEL Chains
# ---------------------------
def get_llm(api_key: str) -> HuggingFaceEndpoint:
    return HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B",
        huggingfacehub_api_token=api_key,
        max_new_tokens=600,
        temperature=0.7
    )

# 1. Prompts
map_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise assistant. Summarize the key points of this chunk."),
    ("human", "Content chunk:\n{text}")
])

combine_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise, helpful assistant that summarizes content clearly."),
    ("human", "Provide a clear, well-structured summary of the following combined summaries in about 300 words.\n\nContent:\n{text}")
])


# ---------------------------
# Button logic
# ---------------------------

if st.button("Summarize the content"):
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("Please provide both a valid NVIDIA API key and a URL.")
    elif not is_valid_url(generic_url):
        st.error("Please enter a valid URL (starting with http or https).")
    else:
        try:
            with st.spinner("Fetching and summarizing content..."):
                # 1. Load documents
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=True,
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        show_progress_bar=True,
                        ssl_verify=False,
                        headers={
                            "User-Agent": (
                                "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/116.0.0.0 Safari/537.36"
                            )
                        },
                    )

                docs = loader.load()

                # 2. Chunk documents using modern text splitter import
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=200,
                )
                split_docs = splitter.split_documents(docs)

                # 3. Build LLM and modern chains via LCEL
                llm = get_llm(hf_api_key)
                
                # Create individual chains
                map_chain = map_prompt | llm | StrOutputParser()
                combine_chain = combine_prompt | llm | StrOutputParser()

                # Execute the "Map" step: summarize each chunk in parallel
                # We extract the page_content string from each Document object
                '''chunk_summaries = map_chain.batch([{"text": doc.page_content} for doc in split_docs],
                                                  config={"max_concurrency": 1} )''' # Change to 2 or 3 if your key supports it
                # Process each chunk sequentially to avoid triggering HF rate limits
                chunk_summaries = []
                for doc in split_docs:
                    summary = map_chain.invoke({"text": doc.page_content},config={"max_concurrency": 1})
                    chunk_summaries.append(summary)

                # Combine the intermediate summaries together into one string
                joined_summaries = "\n\n".join(chunk_summaries)

                # Execute the "Reduce" step: generate the final summary
                output = combine_chain.invoke({"text": joined_summaries})

                st.success(output)

        except Exception as e:
            st.error("Something went wrong while summarizing.")
            st.exception(e)