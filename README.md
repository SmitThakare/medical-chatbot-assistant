
---

# 🩺 Medical Chatbot Assistant (RAG-powered with LangChain + Chainlit)

Welcome to your AI-powered medical assistant! This chatbot uses **Retrieval-Augmented Generation (RAG)** to answer health-related questions by searching through embedded medical literature. Built with LangChain, HuggingFace, FAISS, and Chainlit, it’s optimized for Apple Silicon (M1/M2) and designed for clarity, speed, and educational value.

---

## ⚙️ Architecture Overview

- **Document Ingestion**: PDF medical documents are loaded, split into chunks, embedded using HuggingFace models, and stored in a FAISS vector database.
- **Retrieval-Augmented Generation**: When a user asks a question, the system retrieves relevant chunks and uses a language model to generate a contextual answer.
- **LLM Backend**: Uses `google/flan-t5-base` for factual QA, optimized for M1 Macs via PyTorch MPS.
- **Frontend**: Powered by [Chainlit](https://www.chainlit.io/) for interactive chat interface.

---

## 🚀 Features

- ✅ **M1 Mac Compatible**: Uses Apple Silicon acceleration via PyTorch MPS.
- 📚 **PDF Knowledge Base**: Load and embed scanned or digital PDFs.
- 🔍 **Semantic Search**: FAISS-powered retrieval of relevant medical content.
- 🤖 **LLM Integration**: Uses HuggingFace’s `flan-t5-base` for accurate, context-aware responses.
- 🧠 **Custom Prompting**: Tailored prompt template for medical QA.
- 💬 **Chainlit UI**: Clean, interactive chat experience with source citations.

---

## 🛠️ Setup Instructions

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Add medical PDFs**:
   Place your documents in the `data/` folder.

3. **Build the vector database**:
   ```bash
   python ingest.py
   ```

4. **Run the chatbot**:
   ```bash
   chainlit run model.py
   ```

---

## 💬 Example Questions

- What are the symptoms of diabetes?
- How is hypertension diagnosed?
- What causes migraine headaches?
- What are the side effects of common medications?

---

## 📌 Notes

- ⚠️ **Medical Disclaimer**: This chatbot is for educational purposes only. Always consult a licensed healthcare provider for medical advice or treatment.
- 📚 **Knowledge Base**: Answers are based on embedded PDFs and may not reflect the latest clinical guidelines.
- 🔍 **Source Citations**: Each response includes references to the original documents used.

---

## 🧪 Tech Stack

| Component        | Technology                          |
|------------------|-------------------------------------|
| Embeddings       | `sentence-transformers/all-mpnet-base-v2` |
| LLM              | `google/flan-t5-base`               |
| Vector Store     | FAISS                               |
| Framework        | LangChain                           |
| UI               | Chainlit                            |
| Hardware Support | Apple Silicon (M1/M2) via PyTorch MPS |

---

*Built for clarity, powered by context. Your medical questions—answered with precision.*  
