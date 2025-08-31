# ingest.py - Medical QA Bot Vector DB Builder (M1 Mac Compatible)
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------------- Paths ----------------
DATA_PATH = 'data/'                  # Folder containing PDF(s)
DB_FAISS_PATH = 'vectorstore/db_faiss'  # FAISS DB path

# ---------------- Config ----------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'


# ---------------- Functions ----------------
def load_documents(use_ocr=False):
    """
    Load PDFs from DATA_PATH. If use_ocr=True, attempt OCR extraction for scanned PDFs.
    """
    loader_cls = PyPDFLoader if not use_ocr else UnstructuredPDFLoader
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=loader_cls)
    print(f"📄 Loading documents (OCR={use_ocr})...")
    documents = loader.load()
    print(f"📑 Loaded {len(documents)} documents")
    return documents

def split_documents(documents):
    """
    Split documents into text chunks for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = splitter.split_documents(documents)
    print(f"✂️ Split into {len(texts)} text chunks")
    return texts

def create_embeddings(texts):
    """
    Create HuggingFace embeddings for documents.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    return embeddings

def build_faiss_db(texts, embeddings):
    """
    Build and save FAISS vectorstore.
    """
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"💾 FAISS DB saved at {DB_FAISS_PATH}")
    return db

def create_vector_db(use_ocr=False):
    documents = load_documents(use_ocr=use_ocr)
    if not documents:
        print("⚠️ No documents found. Check your data folder.")
        return
    texts = split_documents(documents)
    embeddings = create_embeddings(texts)
    db = build_faiss_db(texts, embeddings)
    return db

# ---------------- Main ----------------
if __name__ == "__main__":
    # Ensure vectorstore folder exists
    if not os.path.exists('vectorstore'):
        os.makedirs('vectorstore')

    # Try normal text extraction first
    try:
        create_vector_db(use_ocr=False)
    except Exception as e:
        print(f"⚠️ Text extraction failed: {e}")
        print("🔄 Retrying with OCR...")
        create_vector_db(use_ocr=True)