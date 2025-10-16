import streamlit as st
import os
import chromadb
import subprocess
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.retrievers import MultiQueryRetriever
from chromadb.config import Settings

from dotenv import load_dotenv

load_dotenv()

client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chromadb"))
collection = client.get_or_create_collection(name="chroma.sqlite3")

collection.delete(where={"source": "chromadb"})

st.set_page_config(layout="wide")
st.title("RAG Chatbot")

def get_available_models():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')[1:] 
        models = []
        for line in lines:
            if line.strip():
                parts = line.split()
                if parts:
                    model_name = parts[0]
                    models.append(model_name)

        chat_models = []
        for model in models:
            try:
                test_llm = ChatOllama(model=model, temperature=0.1)
                test_response = test_llm.invoke("Hi?")

                if test_response and hasattr(test_response, 'content'):
                    chat_models.append(model)
                    print(f"{model} suppport chat")
                else:
                    print(f"{model} does not support chat")
            except Exception as e:
                print(f"{model} failed testresponse")
                continue
        
        if not chat_models:
            st.warning("No model support chat. Please use available model")
            return models

        return chat_models

    except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
        st.error(f"Error fetching models: {e}. Make sure Ollama is installed and running.")
        return ["gemma3:4b"]

def PDFLoader(pdf_dir):
    all_docs = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, filename)
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                all_docs.extend(docs)
                print(f"Loaded {len(docs)} pages from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return all_docs

def check_database_exists():
    try:
        if os.path.exists(persist_directory):
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            collection_count = vectorstore._collection.count()
            print(f"Database exists with {collection_count} documents")
            return collection_count > 0
        return False
    except Exception as e:
        print(f"Error checking database: {e}")
        return False

st.sidebar.title("Model Selection")

if "available_models" not in st.session_state:
    with st.sidebar:
        with st.spinner("Loading for availavle models:"):
            st.session_state.available_models = get_available_models()

available_models = st.session_state.available_models

selected_model = st.sidebar.selectbox(
    "Select Chat Model",
    options=available_models,
    index=0,
    help="Choose the Ollama model for chat responses"
)

MODEL = selected_model

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state or st.session_state.get("prev_context_size") != 16384:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.session_state.prev_context_size = 16384

llm = ChatOllama(model=MODEL, streaming=True, temperature=0.6)
print(f"selected model of LLM: {llm.model}")

embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")

pdf_files_path = "/Users/User/Desktop/RAG-Chatbot/files/"
persist_directory = "./chromadb"

if "vectorstore" not in st.session_state:
    if check_database_exists():
        with st.sidebar:
            with st.spinner("Loading existing database..."):
                st.session_state.vectorstore = Chroma(
                    persist_directory=persist_directory, 
                    embedding_function=embeddings
                )
        print("Loaded existing ChromaDB database")
    else:
        with st.sidebar:
            with st.spinner("Processing documents and creating database..."):
                all_pdfs = PDFLoader(pdf_files_path)
                
                if not all_pdfs:
                    st.error("No PDF files found in the specified directory")
                    st.stop()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=950,
                    length_function=len,
                    is_separator_regex=False,
                )

                splited_documents = text_splitter.split_documents(all_pdfs)
                print(f"Loaded {len(splited_documents)} document chunks")
                
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=splited_documents,
                    embedding=embeddings,
                    persist_directory=persist_directory
                )
                st.session_state.vectorstore.persist()
                print("Created new ChromaDB database")

vectorstore = st.session_state.vectorstore
print("Data stored in Chromadb")

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate 3
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by new lines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vectorstore.as_retriever(), llm, prompt=QUERY_PROMPT
)

def trim_memory():
    while len(st.session_state.chat_history) > 10 * 2:
        st.session_state.chat_history.pop(0)

def get_chat_context():
    if len(st.session_state.chat_history) >= 2:
        recent_history = st.session_state.chat_history[-2:]
        
        last_user_msg = None
        last_assistant_msg = None
        
        for msg in reversed(st.session_state.chat_history):
            if msg["role"] == "assistant" and last_assistant_msg is None:
                last_assistant_msg = msg["content"]
            elif msg["role"] == "user" and last_user_msg is None:
                last_user_msg = msg["content"]
            
            if last_user_msg and last_assistant_msg:
                break
        
        if last_user_msg and last_assistant_msg:
            return f"\nPrevious Question: {last_user_msg}\nPrevious Answer: {last_assistant_msg}\n"
    
    return ""

def is_follow_up_question(current_question, previous_question):
    if not previous_question:
        return False
    
    strong_indicators = [
        'that', 'this', 'it', 'they', 'them', 
        'more about', 'explain that', 'elaborate on',
        'what about', 'how about', 'and'
    ]
    
    current_lower = current_question.lower()
    return any(indicator in current_lower for indicator in strong_indicators)

custom_prompt_template = """
Answer the question only with the following context and chat history if provided. No need to show your reasoning and thinking process.

Context from documents: {context}

Current Question: {question}

Instructions:
1. Check if the current question might be a follow-up that relates to previous information discussed.
2. If the question seems like a follow-up (e.g., "tell me more", "what about...", "explain that"), use the document context to provide relevant information.
3. If the current question is independent, answer based solely on the document context. If have mention document name, directly find for that document to generate answer.
4. Only use information from the provided context. Do not use your own knowledge or external knowledge.
5. If the context does not contain relevant information to answer the question, respond with: "I could not find relevant information in the provided documents to answer this question."
6. When possible, directly reference relevant parts of the context in your answer.
7. Figures refer to images which should be explained in words when referred to.
8. Assume that all grammar and spelling mistakes in the context are intentional and should not be corrected.
9. Always be truthful and clear while using simple language.

Answer:"""

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=custom_prompt_template,
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask something"):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("Thinking...")
        
        try:
            print("Prompt received:", user_input)            
            chat_context = get_chat_context()
            print("Chat context:", chat_context)

            previous_question = None
            
            if chat_context:
                lines = chat_context.split('\n')
                for line in lines:
                    if line.startswith("Previous Question:"):
                        previous_question = line.replace("Previous Question:", "").strip()
                        break
            
            use_chat_history = False
            augmented_input = user_input
            
            if previous_question and is_follow_up_question(user_input, previous_question):
                use_chat_history = True
                augmented_input = f"{chat_context}\nCurrent question: {user_input}"
                print("Using chat history - detected follow-up question")
            else:
                print("Not using chat history - new topic detected")
            
            retrieve_docs = retriever.invoke(augmented_input)
            print("Documents retrieved:", len(retrieve_docs) if retrieve_docs else 0)
            print("Documents retrieved:", retrieve_docs)

            if not retrieve_docs:
                response = "No relevant documents found."
            else:
                print("Passing to QA chain...")
                doc_context = "\n\n".join([doc.page_content for doc in retrieve_docs])
                
                if use_chat_history:
                    combined_context = f"{chat_context}\n\nDocument Information:\n{doc_context}"
                else:
                    combined_context = doc_context
                
                new_prompt = custom_prompt_template.format(
                    context=combined_context,
                    question=user_input
                )
                
                response = llm.invoke(new_prompt).content
                print("LLM response generated")
                            
            if retrieve_docs:
                sources = set()
                for doc in retrieve_docs:
                    source = doc.metadata.get('source', 'Unknown')
                    if source != 'Unknown':
                        filename = os.path.basename(source)
                        sources.add(filename)
                
                if sources:
                    response += f"\n\n**Source documents:** {', '.join(sorted(sources))}"

            response_placeholder.markdown(response)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            trim_memory()
            
        except Exception as e:
            print("Error during response generation:", e)
            error_response = f"Error: {e}"
            response_placeholder.markdown(error_response)
            st.session_state.chat_history.append({"role": "assistant", "content": error_response})
            
with st.sidebar:
    if st.button("New conversation"):
        st.session_state.memory.clear()
        st.session_state.chat_history = []

        st.rerun()
