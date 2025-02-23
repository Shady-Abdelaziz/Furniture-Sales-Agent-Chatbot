import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langdetect import detect
import os
import pickle
import requests

# Constants
PDF_URL = "https://raw.githubusercontent.com/Shady-Abdelaziz/Sales-Agent/main/1561986011.General%20Catalogue.pdf"
PDF_PATH = "catalog.pdf"
VECTORSTORE_PATH = '.cache/vectorstore.pkl'
DOCS_CACHE_PATH = '.cache/docs_cache.pkl'

# Ensure cache directory exists
os.makedirs('.cache', exist_ok=True)

# Download PDF if not exists
def download_pdf():
    if not os.path.exists(PDF_PATH):
        response = requests.get(PDF_URL)
        with open(PDF_PATH, 'wb') as f:
            f.write(response.content)

@st.cache_resource
def get_embeddings():
    """Initialize and cache embeddings model."""
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        return None

@st.cache_resource
def get_llm():
    """Initialize and cache LLM."""
    try:
        huggingface_api_key = st.secrets["HUGGINGFACE_API_KEY"]
        return HuggingFaceHub(
            repo_id="google/flan-t5-large",
            huggingfacehub_api_token=huggingface_api_key,
            model_kwargs={"temperature": 0.7}
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

def load_or_process_documents():
    """Load documents from cache or process them if cache doesn't exist."""
    try:
        if os.path.exists(DOCS_CACHE_PATH):
            with open(DOCS_CACHE_PATH, 'rb') as f:
                return pickle.load(f)
        
        # Ensure PDF is downloaded
        download_pdf()
        
        loader = PDFPlumberLoader(PDF_PATH)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        documents = text_splitter.split_documents(docs)
        
        with open(DOCS_CACHE_PATH, 'wb') as f:
            pickle.dump(documents, f)
        
        return documents
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return None

def load_or_create_vectorstore(documents, embeddings):
    """Load vectorstore from disk or create if it doesn't exist."""
    try:
        if os.path.exists(VECTORSTORE_PATH):
            with open(VECTORSTORE_PATH, 'rb') as f:
                return pickle.load(f)
        
        if documents is None or embeddings is None:
            return None
            
        vector_store = FAISS.from_documents(documents, embeddings)
        
        with open(VECTORSTORE_PATH, 'wb') as f:
            pickle.dump(vector_store, f)
        
        return vector_store
    except Exception as e:
        st.error(f"Error with vector store: {str(e)}")
        return None

@st.cache_resource
def initialize_system():
    """Initialize the system with caching for better performance."""
    embeddings = get_embeddings()
    if embeddings is None:
        return None
        
    documents = load_or_process_documents()
    if documents is None:
        return None
        
    vector_store = load_or_create_vectorstore(documents, embeddings)
    if vector_store is None:
        return None
        
    retriever = vector_store.as_retriever(search_kwargs={"k": 7})

    prompt_template = """
    You are a warm, helpful, and professional sales assistant specialized in furniture product details.
    Use only the context below to answer the customer's question.
    If the context does not contain relevant information, respond with: "No relevant data found in the context."

    Context: {context}

    Chat History: {chat_history}
    Current Question: {question}

    Your response should:
    1. Be based only on the context provided
    2. Clearly list the products and make suggestions based on the user question
    3. Avoid providing generic responses or asking for more details if the context already contains relevant data

    Please respond in {language}.

    Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question", "language"],
        template=prompt_template
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )

    llm = get_llm()
    if llm is None:
        return None

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        chain_type="stuff"
    )
    
    return qa_chain

def detect_language(text: str) -> str:
    """Detect the language of the input text."""
    try:
        detected = detect(text)
        return "Arabic" if detected == 'ar' else "English"
    except:
        return "English"

def main():
    st.set_page_config(
        page_title="Multilingual Sales Assistant",
        page_icon="🛋️",
        layout="wide"
    )

    st.title("🛋️ Multilingual Furniture Sales Assistant")
    st.write("Ask questions about our furniture products in English or Arabic!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize QA chain with caching
    with st.spinner("Loading the system..."):
        try:
            qa_chain = initialize_system()
            if qa_chain is None:
                st.error("Failed to initialize the system. Please check the logs.")
                st.stop()
        except Exception as e:
            st.error(f"Error initializing system: {str(e)}")
            st.stop()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about our furniture products..."):
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response_language = detect_language(prompt)
                    query_dict = {
                        "question": prompt,
                        "language": response_language
                    }
                    response = qa_chain(query_dict)
                    assistant_response = response['answer'].strip()
                    
                    st.write(assistant_response)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("""
        This sales assistant can help you with:
        - Product information
        - Price inquiries
        - Product comparisons
        - Recommendations
        
        Supports both English and Arabic! 🌐
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
