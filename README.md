# 🔍 BigQA — Retrieval-Augmented Generation

BigQA is a software architecture designed for querying large volumes of textual data. This application implements a **Retrieval-Augmented Generation (RAG)** pipeline, combining semantic document retrieval with Large Language Models (LLMs) to provide precise, context-aware answers.

## 📚 Scientific Foundation

This implementation is based on the reference architecture proposed in the following research papers:

* **Design Principles and a Software Reference Architecture for Big Data Question Answering Systems (2023)**.  
    [Access Paper](https://www.scitepress.org/Link.aspx?doi=10.5220/0011842700003467)
* **BigQA: A Software Reference Architecture for Big Data Question Answering Systems (2024)**.  
    [Access Paper](https://link.springer.com/chapter/10.1007/978-3-031-64748-2_3)

## 🚀 Features

- **RAG Architecture**: Full integration between document retrieval and generative AI.
- **Vector Search**: Uses HuggingFace embeddings (`all-MiniLM-L6-v2`) for semantic similarity search.
- **Streamlit Interface**: An intuitive and responsive web interface for real-time querying.
- **LLM Integration**: Connected via OpenRouter to access state-of-the-art models (e.g., Qwen, Gemini, GPT).
- **Automated Indexing**: Automatic loading and processing of the `Ono-Enzo/Dataset_test` dataset.

## 🛠️ Tech Stack

- **LangChain**: Framework for orchestrating the AI logic.
- **Streamlit**: For the user interface.
- **OpenRouter**: Gateway for LLM access.
- **Hugging Face Datasets & Embeddings**: For data management and vectorization.

## Prerequisites

- Python 3.8+
- An API Key from OpenRouter

## Installation

1. **Clone the repository** (or download the files):
   ```bash
   git clone https://github.com/Enzoonofre/RAG_BigQA.git
   cd RAG_BigQA
   ```

2. **Install dependencies**:
   ```bash
   pip install streamlit python-dotenv langchain-openai datasets langchain-community langchain-huggingface
   ```

## Configuration

1. Create a `.env` file in the root directory of the project.
2. Add your OpenRouter API key to the file:
   ```env
   OPENROUTER_API_KEY=your_api_key_here
   ```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default web browser. It will automatically download the dataset, generate embeddings, and prepare the LLM. Once ready, you can type your questions in the input field.