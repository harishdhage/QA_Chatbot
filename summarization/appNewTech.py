import streamlit as st
from urllib.parse import urlparse

from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize import load_summarize_chain


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
    nvidia_api_key = st.text_input(
        "Enter your NVIDIA API key",
        value="",
        type="password",
    )
    st.markdown(
        "Model: `google/gemma-4-31b-it` via NVIDIA NIM\n\n"
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
# LLM + Prompt
# ---------------------------

# Chat prompt for summarization
summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a concise, helpful assistant that summarizes content clearly.",
        ),
        (
            "human",
            "Provide a clear, well-structured summary of the following content "
            "in about 300 words.\n\nContent:\n{text}",
        ),
    ]
)


def get_llm(api_key: str) -> ChatNVIDIA:
    return ChatNVIDIA(
        model="google/gemma-4-31b-it",
        nvidia_api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1/chat/completions",
        temperature=0.3,
    )


# ---------------------------
# Button logic
# ---------------------------

if st.button("Summarize the content"):
    if not nvidia_api_key.strip() or not generic_url.strip():
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

                # 2. Chunk documents (safer for long pages/videos)
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=200,
                )
                split_docs = splitter.split_documents(docs)

                # 3. Build LLM and chain
                llm = get_llm(nvidia_api_key)

                # map_reduce handles long content better than stuff
                chain = load_summarize_chain(
                    llm,
                    chain_type="map_reduce",
                    combine_prompt=summary_prompt,
                    map_prompt=summary_prompt,
                    verbose=False,
                )

                # 4. Run chain
                output = chain.run(split_docs)

                st.success(output)

        except Exception as e:
            st.error("Something went wrong while summarizing.")
            st.exception(e)
