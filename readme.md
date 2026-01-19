# HM-RAG
HM-RAG is an end-to-end document question-answering system based on Retrieval-Augmented Generation (RAG). It provides a full pipeline from document parsing and vector insertion to retrieval and answer generation.

# 🚀 Overview
HM-RAG integrates several key components to support document-based question answering:
Document parsing and analysis using MinerU.
Graph construction and processing of extracted information.
Vector insertion into databases such as Nebula Graph and Milvus.
Retrieval of relevant content based on user queries.
Answer generation guided by retrieved context.
The system is designed to work with large collections of documents and to support retrieval and reasoning over structured and unstructured knowledge.

# 🧱 Architecture
HM-RAG is composed of the following major parts:
├── DocProcess/           # Document parsing and processing
├── GraphProcess/         # Graph creation and manipulations
├── EntityExtraction/     # Entity detection and Linking
├── Retrieval/            # Vector storage and search
├── Generation/           # LLM-based answer generation
├── MinerU/               # MinerU integration
├── LLM/                  # Large Language Model interfaces
├── Utils/                # Utilities and helpers
├── main.py               # Main entrypoint
├── data/pdf              # Example documents
└── ...

# 📝 Features
End-to-end RAG workflow for document QA
MinerU-based parsing for rich semantic extraction
Support for Nebula Graph and Milvus as vector/graph stores
Retrieval and generation integration

# 📦 Requirements
Before running HM-RAG, you must have:
MinerU environment installed and configured
Nebula Graph service up and running
Milvus vector database running
Python dependencies installed

# 📌 Getting Started
Below is a minimal run example:

# Install dependencies
conda env create -n HM-RAG -f environment.yml

# Run the main pipeline
python main.py