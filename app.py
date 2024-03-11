import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders import WebBaseLoader 
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from pypdf import PdfReader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


def get_text_from_url(url):
    loader = WebBaseLoader(url)
    return loader.load()

def get_text_from_pdf(pdf_file):
    text = ""
    file = PdfReader(pdf_file)
    for page in file.pages:
        text += page.extract_text()
    return text

def get_text_from_txt(txt_file):
    loader = TextLoader(txt_file)
    return loader.load()

def get_text_from_pptx(pptx_file):
    loader = UnstructuredPowerPointLoader(pptx_file)
    return loader.load()

def get_text_from_docx(docx_file):
    loader = Docx2txtLoader(docx_file)
    return loader.load()

def get_text_from_youtube(youtube_url):
    loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
    print("==="*50)
    print(loader.load())
    print("==="*50)
    return loader.load()

def get_text(document):
    if document == None:
        return ""
    elif document.name.endswith("pdf"):
        return get_text_from_pdf(document)
    elif document.name.endswith("docx"):
        return get_text_from_docx(document)
    elif document.name.endswith("pptx"):
        return get_text_from_pptx(document)
    elif document.name.endswith("txt"):
        return get_text_from_txt(document)
    
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
        chunk_size=400, 
        chunk_overlap=40 
    )
    chunks = splitter.split_text(text)
    return chunks

def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def get_context_retriever_chain(vector_store):
    llm = ChatOllama(model="mistral:7b-instruct-q3_K_M")
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatOllama(model="mistral:7b-instruct-q3_K_M")
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']


st.set_page_config(page_title="Advanced RAG", page_icon="🤖")
st.title("Advanced Q&A RAG")


with st.sidebar:
    source = st.selectbox("Choose Knowledge Source", ["select", "Document", "Blog", "YouTube"])
    if source == "select":
        data = None
        pass
    elif source == "Document":
        data = st.file_uploader("Upload Document", type=["pdf", "docx", "pptx", "txt"])
    elif source == "Blog":
        data = st.text_input(f"Enter Blog URL")
    elif source == "YouTube":
        data = st.text_input(f"Enter Youtube URL")
    
    learn = st.button("Read Data")

if learn is None or data is None:
    st.info("Please select a Knowledge Source")
else:
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore(chunk_text(get_text(data)))
    #get_text_from_youtube(data)[0].page_content)
    # if source=="Document":
    #     text = get_text(data)
    # elif source == "Blog":
    #     text = get_text_from_url(data)
    # elif source == "YouTube":
    #     text = get_text_from_youtube(data)
   
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
