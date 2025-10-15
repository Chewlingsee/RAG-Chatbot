# RAG Chatbot with LangChain, Ollama (Deepseek), Streamlit & ChromaDB

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on PDF documents. Built using:

- **LangChain** for document processing and chaining
- **Ollama (Deepseek-r1)** as the local LLM
- **ChromaDB** for storing embeddings
- **Streamlit** for the web interface
- **PDF parsing** via LangChain’s document loaders

---

## Project Overview

This chatbot loads PDFs from a local folder, breaks them into text chunks, embeds them, and stores them in a Chroma vector database. When you ask a question, it retrieves the most relevant chunks and passes them into a local LLM (Deepseek via Ollama) to generate a context-aware response.

---

## Project Structure

```

rag-chatbot/
├── app.py                  # Streamlit app UI, vectorstore and QA chain, loads and processes PDF files
├── files/                    # Folder containing PDF documents
├── chromadb/               # Local ChromaDB vectorstore
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

````

---

## Installation & Setup

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Git and pip

### 2. Clone the Repository

```bash
git clone https://github.com/rw-xzer/rag-chatbot.git
cd rag-chatbot
````

### 3. Set Up the Environment

```bash
python -m venv venv
venv\Scripts\activate      # On Mac: venv/bin/activate
pip install -r requirements.txt
```

### 4. Pull the Deepseek Model

```bash
ollama pull deepseek
```

---

## Running the App

Make sure Ollama is running, then start the Streamlit app:

```bash
streamlit run app.py
```

The app will load PDFs from the `files/` folder and use them to answer your questions.

---

## How It Works

1. **Document Loading**
   PDFs are loaded using `FileSystemBlobLoader` and parsed with `PyPDFParser`.

2. **Text Splitting**
   Text is split into chunks using `RecursiveCharacterTextSplitter` for better embedding and retrieval.

3. **Embedding**
   Chunks are embedded using `OllamaEmbeddings`.

4. **Vector Store**
   Embeddings are stored in a local ChromaDB instance (`chroma_db/`).

5. **Retrieval**
   `MultiQueryRetriever` fetches relevant chunks for the user’s question.

6. **LLM Response**
   `ChatOllama` (running Deepseek) uses the context to generate a response.

7. **UI**
   A Streamlit interface takes user input and displays the output.

---

## requirements.txt

```txt
streamlit
langchain
langchain_ollama
langchain_community
langchain_text_splitters
chromadb
pypdf
```

---

## Potential Improvements

* Allow PDF uploads directly from the UI
* Show source chunks used in the answer
* Switch to a cloud-hosted vector DB
* Add more model options via Ollama

---

## Author

* **GitHub:** rw-xzer
* **Date:** July 2025

---

## Notes

This is a minimal prototype built for learning and experimentation. Open for contributions, forks and improvements.
