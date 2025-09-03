# RAG-implementation
RAG Implementation using LangChain

This project demonstrates how to build a basic Retrieval-Augmented Generation (RAG) pipeline leveraging LangChain, Google Generative AI's Gemini-2.0 model, and ChromaDB. The pipeline loads, processes, embeds, and indexes PDF documents for retrieval-augmented question answering.

Table of Contents
Overview

Features

Tech Stack

Setup Instructions

Workflow Details

Example Usage

Notes

Overview
The notebook implements a pipeline where a user’s question is answered using information retrieved from a PDF document.
It loads a PDF from a URL, splits the content into manageable chunks, creates vector embeddings, stores them in a vectorstore (Chroma), and then uses Gemini-2.0 via LangChain for retrieval-augmented QA.

Features
PDF loading from URL

Splitting documents into overlapping text chunks

Creating embeddings using Google Generative AI

Vector storage and semantic search with ChromaDB

Retrieval-Augmented QA using Gemini-2.0 (LangChain integration)

Example Q&A workflow

Tech Stack
Python (Colab/Jupyter)

LangChain

langchain-google-genai

ChromaDB

pypdf

Google Generative AI (Gemini-2.0/Flash Model)

Setup Instructions
In a Colab/Jupyter notebook:

Install Required Packages

python
!pip install -q --upgrade google-generativeai langchain-google-genai
!pip install langchain_community
!pip install pypdf
!pip install chromadb
API Key Setup
Ensure your Google Generative AI API Key is stored in Colab’s userdata as GOOGLE_API_KEY:

python
from google.colab import userdata
API_KEY = userdata.get('GOOGLE_API_KEY')
Workflow Details
Model Initialization
Create a Gemini-2.0 Flash chat model using your API key.

python
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=API_KEY)
Document Loading
Load a PDF document from the web.

python
from langchain_community.document_loaders import PyPDFLoader
pdfloader = PyPDFLoader('<PDF_URL>')
loaded_pdf_doc = pdfloader.load()
Splitting Text
Break the loaded document into text chunks for embedding.

python
from langchain.text_splitter import RecursiveCharacterTextSplitter
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, separators=['\n\n', '\n', ' '])
chunks = recursive_splitter.split_documents(loaded_pdf_doc)
Embedding Creation
Generate embeddings for each chunk using Google Generative AI.

python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=API_KEY)
Vector Storage
Store the embeddings using Chroma as a vector database.

python
from langchain.vectorstores import Chroma
vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
Retriever Configuration
Build a retriever interface for similarity search.

python
vector_index = vector_store.as_retriever(search_kwargs={'k':8})
Setup RetrievalQA Chain
Compose a RAG pipeline using LangChain’s RetrievalQA.

python
from langchain.chains import RetrievalQA
rag = RetrievalQA.from_chain_type(
    model,
    retriever = vector_index,
    return_source_documents=True
)
Ask Questions
Run questions through the RAG system.

python
question = 'Can you talk about what is there in property1?'
response = rag({"query": question})
print(response)
Example Usage
python
# Assuming all steps above are executed
question = 'Can you talk about what is there in property1?'
response = rag({"query": question})
print(response)
This will output the answer along with the supporting source documents from the indexed PDF.

Notes
Replace <PDF_URL> with the actual URL of your target PDF.

Make sure your API key quota is sufficient for your usage.

To load local PDFs, use the appropriate local file path with PyPDFLoader.

You can customize the chunk_size, overlap, retriever parameters, or the Gemini model version.

Chroma runs locally by default in Colab/Jupyter. For larger-scale or persistent storage, refer to Chroma documentation for alternative setups.

End of README
