import os
import time
import chromadb
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from typing import Any, List, Generator, AsyncGenerator
from llama_parse import LlamaParse
from llama_index.core.llms import LLM, ChatMessage, ChatResponse
import re
import aiohttp
import pyperclip

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.makedirs("files", exist_ok=True)
os.makedirs("chroma_db", exist_ok=True)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

llama_parse_api_key = os.getenv("LLAMA_PARSE_API_KEY")
if not llama_parse_api_key:
    st.error("LlamaParse API key not found. Please set it in the environment variables.")
    st.stop()
parser = LlamaParse(api_key=llama_parse_api_key)

class OllamaLLM(LLM):
    model: str = "mistral"  

    @property
    def metadata(self):
        return {"model": self.model}

    def complete(self, prompt: str) -> str:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False}
        )
        if response.status_code == 200:
            return response.json().get("response", "No response").strip()
        return "Error from Ollama."

    def stream_complete(self, prompt: str) -> Generator[str, None, None]:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": True},
            stream=True
        )
        for line in response.iter_lines():
            if line:
                yield line.decode('utf-8')

    def chat(self, messages: List[ChatMessage]) -> ChatResponse:
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in messages]) + "\nAssistant:"
        response = self.complete(prompt)
        return ChatResponse(message=ChatMessage(role="assistant", content=response))

    def stream_chat(self, messages: List[ChatMessage]) -> Generator[ChatResponse, None, None]:
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in messages]) + "\nAssistant:"
        for token in self.stream_complete(prompt):
            yield ChatResponse(message=ChatMessage(role="assistant", content=token))

    async def acomplete(self, prompt: str) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("response", "No response").strip()
                return "Error from Ollama."

    async def astream_complete(self, prompt: str) -> AsyncGenerator[str, None]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": True}
            ) as response:
                async for line in response.content:
                    if line:
                        yield line.decode('utf-8')

    async def achat(self, messages: List[ChatMessage]) -> ChatResponse:
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in messages]) + "\nAssistant:"
        response = await self.acomplete(prompt)
        return ChatResponse(message=ChatMessage(role="assistant", content=response))

    async def astream_chat(self, messages: List[ChatMessage]) -> AsyncGenerator[ChatResponse, None]:
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in messages]) + "\nAssistant:"
        async for token in self.astream_complete(prompt):
            yield ChatResponse(message=ChatMessage(role="assistant", content=token))

Settings.llm = OllamaLLM()

if "index" not in st.session_state:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("ragbot")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    st.session_state.index = VectorStoreIndex.from_vector_store(vector_store)

def render_dashboard(df, file_name):
    st.subheader(f"Data Preview for {file_name}")
    st.write(df.head())
    st.subheader("Data Overview")
    st.write(df.describe())
    st.subheader("Data Visualizations")
    if st.checkbox("Show Correlation Heatmap"):
        st.write("### Correlation Heatmap")
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) >= 2:
        st.write("### Select Columns for Scatter Plot")
        x_col = st.selectbox("Select X-axis Column", numeric_cols)
        y_col = st.selectbox("Select Y-axis Column", numeric_cols)
        if x_col and y_col:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=df[x_col], y=df[y_col], color="blue")
            st.pyplot(plt)
    if st.checkbox("Show Histogram"):
        st.write("### Select Column for Histogram")
        hist_col = st.selectbox("Select Column for Histogram", numeric_cols)
        plt.figure(figsize=(8, 6))
        sns.histplot(df[hist_col], kde=True, color="green")
        st.pyplot(plt)
    st.subheader("AI-Insights")
    st.write("Analyzing trends using basic AI logic.")
    for col in numeric_cols:
        st.write(f"**{col}**")
        st.write(f" - Mean: {df[col].mean()}")
        st.write(f" - Median: {df[col].median()}")
        st.write(f" - Std Dev: {df[col].std()}")

st.set_page_config(page_title="RAGbot", layout="centered")
st.title("🤖 RAGbot")
st.caption("Chat and attach files to query")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

with st.sidebar:
    st.write("Chat History")
    st.button("New Chat", key="new_chat", on_click=lambda: [st.session_state.chat_history.clear(), st.session_state.processed_files.clear()])
    for i, msg in enumerate(st.session_state.chat_history):
        with st.expander(f"{msg['role'].capitalize()}: {msg['message'][:30]}..."):
            st.markdown(msg["message"])
            st.button("Copy", key=f"copy_history_{i}", on_click=lambda x=msg["message"]: pyperclip.copy(x))

chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ddd; padding: 10px; background-color: #000000; color: #ffffff; font-family: monospace;">
                    {msg['message']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.button("Copy", key=f"copy_{msg['message'][:10]}_{time.time()}", on_click=lambda x=msg["message"]: pyperclip.copy(x))
            else:
                st.markdown(msg["message"])

user_input = st.chat_input("Ask a question or include a URL to process:")
uploaded_files = st.file_uploader("📎 Attach Files", accept_multiple_files=True, label_visibility="collapsed")

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join("files", uploaded_file.name)
        file_key = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        if file_key not in st.session_state.processed_files:
            if file_path.endswith(".csv"):
                csv_df = pd.read_csv(file_path)
                if st.checkbox(f"Generate AI Dashboard for {file_key}"):
                    render_dashboard(csv_df, file_key)
                else:
                    with st.status(f"Processing {file_key}..."):
                        parsed_data = parser.load_data(file_path)
                        for item in parsed_data:
                            doc = Document(text=item.text, metadata={"source": file_path})
                            st.session_state.index.insert(doc)
                        st.session_state.processed_files.add(file_key)
                    st.success(f"✅ {file_key} processed and stored.")
            else:
                with st.status(f"Processing {file_key}..."):
                    parsed_data = parser.load_data(file_path)
                    for item in parsed_data:
                        doc = Document(text=item.text, metadata={"source": file_path})
                        st.session_state.index.insert(doc)
                    st.session_state.processed_files.add(file_key)
                st.success(f"✅ {file_key} processed and stored.")

if user_input:
    st.session_state.pending_prompt = user_input
    with chat_container:
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Assistant is thinking..."):
                url_pattern = r'(https?://\S+)'
                urls = re.findall(url_pattern, user_input)
                for url in urls:
                    if url not in st.session_state.processed_files:
                        with st.status(f"Fetching content from {url}..."):
                            doc = Document(text=requests.get(url).text, metadata={"source": url})
                            st.session_state.index.insert(doc)
                            st.session_state.processed_files.add(url)
                        st.write(f"- {url} (URL processed)")
                retriever = st.session_state.index.as_retriever(similarity_top_k=3)
                retrieved_nodes = retriever.retrieve(user_input)
                context_str = "\n\n".join([node.text for node in retrieved_nodes])
                previous_qna = "\n\n".join(
                    [f"{m['role'].capitalize()}: {m['message']}" for m in st.session_state.chat_history[-6:]]
                )
                prompt = f"""You are a helpful assistant answering questions from document/image/web/csv content.

Context:
{context_str}

Conversation History:
{previous_qna}

User: {user_input}
Assistant:"""
                response = Settings.llm.complete(prompt)
                placeholder = st.empty()
                streamed = ""
                for word in response.split():
                    streamed += word + " "
                    placeholder.markdown(
                        f"""
                        <div style="border: 1px solid #ddd; padding: 10px; background-color: #000000; color: #ffffff; font-family: monospace;">
                        {streamed}▌
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    time.sleep(0.03)
                placeholder.markdown(
                    f"""
                    <div style="border: 1px solid #ddd; padding: 10px; background-color: #000000; color: #ffffff; font-family: monospace;">
                    {streamed}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.button("Copy", key=f"copy_response_{time.time()}", on_click=lambda x=streamed: pyperclip.copy(x))
                st.session_state.chat_history.append({"role": "assistant", "message": streamed})
    st.session_state.pending_prompt = None
