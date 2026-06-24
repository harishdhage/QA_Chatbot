import streamlit as st
from typing import Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_classic.chains.summarize import load_summarize_chain 
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from urllib.parse import urlparse


def create_retry_session(timeout: float = 60.0, retries: int = 2, backoff_factor: float = 0.5) -> requests.Session:
    session = requests.Session()
    session.verify = True
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    original_request = session.request

    def request_with_timeout(method: str, url: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", timeout)
        return original_request(method, url, **kwargs)

    session.request = request_with_timeout  # type: ignore[assignment]
    return session

# UI field creation
st.set_page_config(page_title="Lanchain: Summarize the content from any URL and Youtube video ", page_icon="🦈")

def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)

st.title("🦈 Lanchain: Summarize the content from any URL and Youtube video 🦈")
st.subheader("Summarize the URL")


# Get api key
with st.sidebar:
    nvidia_api_key = st.text_input("Enter the NVIDIA api key",value="",type="password")

llm = ChatNVIDIA(model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
                 nvidia_api_key=nvidia_api_key,
                 base_url="https://integrate.api.nvidia.com/v1")
llm._client.get_session_fn = lambda: create_retry_session(timeout=60.0)
llm._async_client.get_async_session_fn = lambda: create_retry_session(timeout=60.0)

prompt_template = """
Provide the summary for following content in 300 words.
Content:{text}
"""

prompt = PromptTemplate(template=prompt_template,input_variables=["text"])

generic_url=st.text_input("Enter the URL to be summarized",value="",label_visibility="collapsed")

def safe_load_url(url: str):
    try:
        if "youtube.com" in url or "youtu.be" in url:
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        else:
            loader = UnstructuredURLLoader(
                urls=[url],
                ssl_verify=False,
                headers={"User-Agent": "Mozilla/5.0"},
            )
        return loader.load()
    except Exception as e:
        return None


if st.button("Summarize the content from entered Youtube or URL"):
    if not nvidia_api_key.strip() or not generic_url.strip():
        st.error("Enter the correct information!!!")
    elif not is_valid_url(generic_url):
        st.error("Enter the valid url!!!")
    else:
        '''try:
            with st.spinner("waiting ...."):
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url,
                                                            add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url],
                                                   #show_progress_bar=True,
                                                   ssl_verify=False,
                                                   headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    
                doc = loader.load()'''
        docs = safe_load_url(generic_url)

        if not docs:
            st.error("Unable to load content from this URL. The site may block scraping.")
            st.stop()


            ## Chain for summarization
            chain = load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
            output_response = chain.run(doc)

            st.success(output_response)
        '''except Exception as e:
            err_text = str(e)
            if "504" in err_text or "Gateway Timeout" in err_text:
                st.error(
                    "The NVIDIA API returned a gateway timeout. "
                    "Please retry after a few minutes or try a smaller URL/video input."
                )
            else:
                st.error(f"Exceptions : {e}")
                st.exception(e)'''
