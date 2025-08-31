# model.py - Fixed for M1 Mac + Chainlit
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.chains import RetrievalQA
import chainlit as cl
from transformers import AutoModelForSeq2SeqLM

from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

DB_FAISS_PATH = 'vectorstore/db_faiss'

# ---------------- Prompt ----------------
custom_prompt_template = """You are a medical AI assistant. Based on the medical context provided, answer the user's question clearly and accurately.

Context: {context}

Question: {question}

Answer:"""

def set_custom_prompt():
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )

# ---------------- LLM Loader ----------------
def load_llm():
    model_name = "google/flan-t5-base"

    try:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"🚀 Using {'MPS' if device.type == 'mps' else 'CPU'} acceleration")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)  # ✅ Use correct loader

        hf_pipeline = pipeline(
            "text2text-generation",  # ✅ Correct task for T5
            model=model,
            tokenizer=tokenizer,
            device=-1,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=False
        )

        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        print(f"✅ Loaded {model_name}")
        return llm

    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        raise

# ---------------- Retrieval QA ----------------
def retrieval_qa_chain(llm, prompt, db):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 5}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

def qa_bot():
    try:
        print("📚 Loading embeddings...")
        embeddings = HuggingFaceEmbeddings(
             model_name="sentence-transformers/all-mpnet-base-v2",
             model_kwargs={'device': 'cpu'}  # or 'mps' if you're using Apple Silicon
                )


        print("🔍 Loading FAISS DB...")
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

        print("🤖 Loading LLM...")
        llm = load_llm()

        print("⚙️ Building QA chain...")
        qa_prompt = set_custom_prompt()
        qa = retrieval_qa_chain(llm, qa_prompt, db)

        print("✅ QA chain ready")
        return qa

    except Exception as e:
        print(f"❌ QA bot init error: {e}")
        return None

# ---------------- Chainlit ----------------
@cl.on_chat_start
async def start():
    chain = qa_bot()
    if chain is None:
        await cl.Message(content="❌ Failed to initialize QA bot. Check logs.").send()
        return

    cl.user_session.set("chain", chain)

    await cl.Message(
        content="👋 Hi! I’m your Medical Bot. Ask me any medical question based on my knowledge base."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    if chain is None:
        await cl.Message(content="❌ QA chain not available. Restart chat.").send()
        return

    user_q = message.content
    print(f"📝 User asked: {user_q}")

    waiting_msg = await cl.Message(content="🔍 Searching medical knowledge base...").send()

    try:
        # For compatibility: RetrievalQA uses `input` in newer LangChain
        try:
            res = chain.invoke({"input": user_q})
        except Exception:
            res = chain.invoke({"query": user_q})

        raw_answer = res.get("result", "")
        sources = res.get("source_documents", [])
        # 🔍 Debug: Preview source documents
        for i, doc in enumerate(sources[:3]):
            print(f"Source {i+1}: {doc.metadata.get('source')}")
            print(doc.page_content[:300])  # Preview first 300 characters
        answer = raw_answer.strip() if raw_answer else "I couldn’t find a clear answer in the knowledge base."

        response = f"💡 **Medical Information:**\n{answer}"
        if sources:
            response += "\n\n📚 **Sources:**"
            for i, s in enumerate(sources[:3]):
                src = s.metadata.get("source", "Medical literature")
                response += f"\n{i+1}. {src.split('/')[-1]}"

        await cl.Message(content=response).send()

    except Exception as e:
        err = f"❌ Error answering your question: {e}"
        print(err)
        await cl.Message(content=err).send()

if __name__ == "__main__":
    print("🧪 Testing bot init...")
    bot = qa_bot()
    if bot:
        print("✅ Bot initialized OK")
    else:
        print("❌ Bot init failed")