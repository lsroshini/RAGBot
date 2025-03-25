# RAGBot
RAGBot is an AI-powered chatbot that answers questions about various document formats using the concept of Retrieval-Augmented Generation (RAG). It supports PDFs, DOCX, CSV files, images, and URLs. RAGBot extracts and retrieves relevant information using FAISS for similarity search and provides accurate answers.
## 🔎 Setup and Installation  

1. **Install Ollama:**  
   Download and install Ollama from [Ollama's official website](https://ollama.com/download).  
   ```bash
   ollama run mistral
   ```
   This pulls the Mistral LLM for local use.  

2. **Clone the Repository:**  
   ```bash
   git clone https://github.com/YourUsername/RAGBot.git
   cd RAGBot
   ```

3. **Install Dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**  
   ```bash
   streamlit run app.py
   ```

---
