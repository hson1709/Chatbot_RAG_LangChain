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

You can easily run this project using Docker. Make sure you have [Docker](https://www.docker.com/) installed on your system.

### 1. Build the Docker Image

Open a terminal in the project directory and run:

```bash
docker build -t medical-rag-chatbot .
```

### 2. Prepare Environment Variables

Create a `.env` file in the project root with your Google API key:

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

### 3. Run the Container

Start the chatbot with:

```bash
docker run -p 5000:5000 --env-file .env medical-rag-chatbot
```

- The app will be available at [http://localhost:5000](http://localhost:5000).

**Note:**  
Before running the chatbot, you may need to build the vector database inside the container. You can do this by running:

```bash
docker run --env-file .env medical-rag-chatbot python create_db.py
```
