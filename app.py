import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import time
import fitz  
import faiss
import pickle
import pytesseract
import requests
import numpy as np
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from sentence_transformers import SentenceTransformer
import streamlit as st
from docx import Document  
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
VECTOR_DIM = 384
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
INDEX_FILE = "data/faiss_index.pkl"
DOC_EMBED_FILE = "data/docs.pkl"
os.makedirs("files", exist_ok=True)
os.makedirs("data", exist_ok=True)
#Document Ingestion Pipeline
if os.path.exists(INDEX_FILE) and os.path.exists(DOC_EMBED_FILE):
    with open(INDEX_FILE, "rb") as f:
        index = pickle.load(f)
    with open(DOC_EMBED_FILE, "rb") as f:
        documents = pickle.load(f)
else:
    index = faiss.IndexFlatL2(VECTOR_DIM)
    documents = []

def extract_text_chunks(text, chunk_size=1500, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def embed_texts(texts):
    return EMBEDDING_MODEL.encode(texts).tolist()

def persist_index():
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(index, f)
    with open(DOC_EMBED_FILE, "wb") as f:
        pickle.dump(documents, f)

def ocr_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

def fetch_url_text(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        return soup.get_text()
    except Exception as e:
        return f"Failed to fetch URL: {e}"

def extract_docx_text(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    return df

def process_csv_for_query(df):
    text = df.to_string()
    csv_chunks = extract_text_chunks(text)
    csv_embeds = embed_texts(csv_chunks)
    index.add(np.array(csv_embeds).astype("float32"))
    documents.extend(csv_chunks)
    persist_index()

#Query Pipeline
def query_ollama(prompt, model="mistral"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        if response.status_code == 200:
            return response.json().get("response", "No response").strip()
        return "Error from Ollama."
    except Exception as e:
        return f"Ollama error: {e}"

# Dashboard Function
def render_dashboard(df):
    st.subheader("Data Preview")
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
st.caption("Upload and query PDFs, DOCX, Images, URLs, or CSVs")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["message"])


file_type = st.selectbox("Choose Input Type", ["PDF", "DOCX", "Image", "URL", "CSV"])
uploaded_file = None
url_input = ""
csv_df = None

if file_type == "PDF":
    uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf")
elif file_type == "DOCX":
    uploaded_file = st.file_uploader("📜 Upload DOCX", type="docx")
elif file_type == "Image":
    uploaded_file = st.file_uploader("🖼️ Upload Image", type=["png", "jpg", "jpeg"])
elif file_type == "URL":
    url_input = st.text_input("🌐 Enter URL to extract content")
elif file_type == "CSV":
    uploaded_file = st.file_uploader("📊 Upload CSV", type="csv")
    if uploaded_file:
        csv_df = load_csv(uploaded_file)
        st.success("✅ CSV Loaded.")
        if st.checkbox("Generate AI Dashboard"):
            render_dashboard(csv_df)
        else:
            process_csv_for_query(csv_df)

if uploaded_file and file_type != "CSV":
    file_path = os.path.join("files", uploaded_file.name)
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        if file_type == "PDF":
            with st.status("Processing PDF..."):
                doc = fitz.open(file_path)
                pdf_text = "\n".join(page.get_text() for page in doc)
                pdf_chunks = extract_text_chunks(pdf_text)
                pdf_embeds = embed_texts(pdf_chunks)
                index.add(np.array(pdf_embeds).astype("float32"))
                documents.extend(pdf_chunks)
                persist_index()
            st.success("✅ PDF processed.")

        elif file_type == "DOCX":
            with st.status("Processing DOCX..."):
                docx_text = extract_docx_text(file_path)
                if docx_text.strip():
                    docx_chunks = extract_text_chunks(docx_text)
                    docx_embeds = embed_texts(docx_chunks)
                    index.add(np.array(docx_embeds).astype("float32"))
                    documents.extend(docx_chunks)
                    persist_index()
                else:
                    st.warning("❌ No readable text found in DOCX.")
            st.success("✅ DOCX processed.")

        elif file_type == "Image":
            with st.status("Extracting text from image..."):
                image_text = ocr_image(uploaded_file)
                if image_text.strip():
                    image_chunks = extract_text_chunks(image_text)
                    image_embeds = embed_texts(image_chunks)
                    index.add(np.array(image_embeds).astype("float32"))
                    documents.extend(image_chunks)
                    persist_index()
                else:
                    st.warning("❌ No readable text found in image.")
            st.success("✅ Image text extracted.")
    else:
        st.info(f"📌 This {file_type} was already processed.")

if url_input:
    with st.status("Fetching content from URL..."):
        url_text = fetch_url_text(url_input)
        if url_text.strip():
            url_chunks = extract_text_chunks(url_text)
            url_embeds = embed_texts(url_chunks)
            index.add(np.array(url_embeds).astype("float32"))
            documents.extend(url_chunks)
            persist_index()
        else:
            st.warning("❌ No text found at the URL.")
    st.success("✅ URL content added.")


if user_input := st.chat_input(f"Ask a question about your {file_type} (or CSV if not dashboard):"):
    st.session_state.chat_history.append({"role": "user", "message": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Assistant is thinking..."):
            if len(documents) == 0:
                st.warning(f"⚠️ Please upload a {file_type} or process a CSV for querying before asking.")
            else:
                query_vec = EMBEDDING_MODEL.encode([user_input])
                D, I = index.search(np.array(query_vec).astype("float32"), k=3)

                top_chunks = [documents[i] for i in I[0] if i < len(documents)]
                context = "\n\n".join(top_chunks)

                previous_qna = "\n".join(
                    [f"User: {m['message']}" if m["role"] == "user" else f"Assistant: {m['message']}"
                     for m in st.session_state.chat_history[-6:]]
                )
                prompt = f"""You are a helpful assistant answering questions from document/image/web/csv content.

Context:
{context}

Conversation History:
{previous_qna}

User: {user_input}
Assistant:"""

                response = query_ollama(prompt)

                placeholder = st.empty()
                streamed = ""
                for word in response.split():
                    streamed += word + " "
                    placeholder.markdown(streamed + "▌")
                    time.sleep(0.03)
                placeholder.markdown(streamed)

                st.session_state.chat_history.append({"role": "assistant", "message": streamed})