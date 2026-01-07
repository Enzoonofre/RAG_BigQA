import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from datasets import load_dataset
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.messages import HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ----------------------
# Configurações do Streamlit
# ----------------------
st.set_page_config(page_title="BigQA RAG", layout="centered")

primaryColor = "#afbac2"
backgroundColor = "#3d4850"
secondaryBackgroundColor = "#081310"
textColor = "#f5eff8"

st.markdown(
    f"""
    <style>
        /* Cores do tema */
        :root {{
            --primary-color: {primaryColor};
            --background-color: {backgroundColor};
            --secondary-background-color: {secondaryBackgroundColor};
            --text-color: {textColor};
        }}
        /* Aumentar largura do texto e input */
        .bigqa-container {{
            max-width: 1200px;
            margin: auto;
        }}
        .stTextInput>div>div>input {{
            height: 50px;
            font-size: 18px;
        }}
        .stTextArea>div>textarea {{
            font-size: 16px;
            min-height: 120px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Centralizando logo (substitua pelo seu logo)
# _, center_col, _ = st.columns(3)
# with center_col:
#     st.image("assets/bigqa-logo.png", use_column_width=True)

# Título e descrição
st.subheader("🔍 BigQA — Retrieval-Augmented Generation")
st.caption(
    """BigQA é uma arquitetura para consultas em grandes volumes de dados textuais. 
    Ele combina RAG (Retrieval-Augmented Generation) e LLM para gerar respostas contextuais a partir de bases de documentos."""
)

# ----------------------
# Funções de carregamento e indexação
# ----------------------
@st.cache_data(show_spinner=False)
def load_documents():
    st.caption("Fetching dataset 'Ono-Enzo/Dataset_test'")
    dataset = load_dataset("Ono-Enzo/Dataset_test", split="train")
    data = dataset.to_pandas()
    loader = DataFrameLoader(data, page_content_column="chunk")
    documents = loader.load()
    return documents

@st.cache_resource(show_spinner=False)
def get_vectorstore(documents):
    st.caption("Creating embeddings and indexing documents...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device":"cpu"})
    vectorstore = InMemoryVectorStore.from_documents(documents=documents, embedding=embeddings)
    return vectorstore

@st.cache_resource(show_spinner=False)
def get_llm():
    st.caption("Setting up language model...")
    chat = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="openai/gpt-oss-20b:free",
        temperature=0
    )
    return chat

# ----------------------
# Carregamento com status visual
# ----------------------
with st.status("Downloading dataset...", expanded=True) as status:
    documents = load_documents()
    status.update(label=f"Dataset loaded ({len(documents)} documents). Indexing...")
    vectorstore = get_vectorstore(documents)
    status.update(label="Vectorstore ready. Loading LLM...")
    chat = get_llm()
    status.update(label="Ready to answer questions!", state="complete", expanded=False)

# ----------------------
# Interface de pergunta
# ----------------------
st.subheader("Ask your question about BigQA")
st.caption("Type your question and get a context-aware answer generated using RAG.")

if user_query := st.text_input(
    label="Your question:",
    placeholder="Ex: O que é BigQA?",
):
    with st.spinner("Generating answer..."):
        try:
            # Busca nos documentos mais relevantes
            results = vectorstore.similarity_search(user_query, k=3)
            context = "\n".join([x.page_content for x in results])
            prompt = HumanMessage(content=f"Use o contexto abaixo para responder a pergunta.Se responder sem usar o contexto, avise o usuário\n\nContexto:\n{context}\n\nPergunta: {user_query}")
            res = chat.invoke([prompt])

            # Exibição do resultado
            st.success(res.content)

        except Exception as e:
            st.error(f"We could not find an answer for your question. Error: {e}")
