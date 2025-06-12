# Medical RAG ChatBot with LangChain

![image](https://github.com/hson1709/Chatbot_RAG_LangChain/blob/master/UI.png)

## Introduction

This project is a Retrieval-Augmented Generation (RAG) chatbot specialized in the medical domain. It leverages **LangChain**, **Flask**, and the **Google Gemini API** to provide informative responses about diseases and medications.

### Key Features

- Retrieves and extracts relevant medical information from a curated knowledge base.
- Utilizes **ChromaDB** to store and manage vector embeddings.
- Applies a **Cross-Encoder Reranker** for enhanced search accuracy.
- Automatically classifies queries and routes them to the appropriate data source.
- Generates context-aware answers using **Google Gemini**.
- Offers a clean and interactive web interface.

## Models Used

- **Embedding Model:** `all-mpnet-base-v2`
- **Reranking Model:** `ms-marco-MiniLM-L-6-v2`
- **LLM:** `gemini-2.0-pro-exp`

## Data Sources

- **Medications:** Scraped from [Nhà thuốc Long Châu](https://nhathuoclongchau.com.vn)
- **Diseases:** Scraped from [Vinmec](https://www.vinmec.com/vie/tra-cuu-benh/)

### Data Fields

- **Medication Data:** Name, URL, Ingredients, Usage, Indications, Side Effects, Warnings, Storage Instructions, Drug Category.
- **Disease Data:** Causes, Symptoms, Risk Factors, Prevention, Diagnosis, Treatment.

## Setup & Usage

### 1. Environment Setup

Ensure Python 3.8+ is installed.

```bash
pip install -r requirements.txt
```

### 2. Add Google API Key
Create a .env file in the project root:

```bash
GOOGLE_API_KEY = "YOUR API KEY"  
```

### 3. Build Vector Database
Generate the vector database by running:

```python
python create_db.py
```

### 4. Run the Chatbot
Start the web-based chatbot application with:

```python
python app.py
```
