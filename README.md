# RAG Chatbot

## Description

RAG Chatbot is a document-based chatbot that answers user questions using information retrieved from PDF documents. The application loads PDF files from a local folder, stores their embeddings in ChromaDB, retrieves relevant document content, and generates answers using a local Ollama language model.

## Features

- Load PDF documents from a local folder
- Split PDF content into smaller text chunks
- Store document embeddings in ChromaDB
- Retrieve relevant document context based on user questions
- Generate answers using Ollama chat models
- Select available Ollama models from the sidebar
- Display source document names used for the answer
- Maintain simple chat history for follow-up questions

## Required Installation

Make sure you have installed:

- Python 3.10 or above
- Ollama
- Streamlit
- LangChain
- ChromaDB
- Required Ollama models

### Install Ollama Models

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text:v1.5
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

### Install Python Packages

```bash
pip install streamlit chromadb langchain langchain-community langchain-ollama langchain-text-splitters python-dotenv pypdf
```

## How to Run

Create a folder named `files` and place your PDF documents inside it.

```text
RAG-Chatbot/
├── app.py
├── files/
│   └── your-document.pdf
└── chromadb/
```

In the code, make sure the PDF path is correct:

```python
pdf_files_path = "./files/"
```

Start Ollama:

```bash
ollama serve
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Open the app in your browser:

```text
http://localhost:8501
```

## Architecture

```text
PDF Documents
      ↓
PDF Loader
      ↓
Text Splitter
      ↓
Ollama Embeddings
      ↓
ChromaDB Vector Database
      ↓
Retriever / MultiQueryRetriever
      ↓
Relevant Document Context
      ↓
Ollama Chat Model
      ↓
Generated Answer + Source Documents
```

## Notes

The `chromadb` folder is created automatically after the PDF documents are processed.

To reset the database, stop the app, delete the `chromadb` folder, and run the app again.
