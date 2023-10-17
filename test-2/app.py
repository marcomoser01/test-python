from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from glob import glob
from tqdm import tqdm

def load_documents(directory : str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 48)
    documents = []
    for item_path in glob(directory + ".pdf"):
        loader = PyPDFLoader(item_path)
        documents.extend(loader.load_and_split(text_splitter=text_splitter))
        
    return documents

documents = load_documents("data/")

embeddings = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs = {'device': 'cuda'})