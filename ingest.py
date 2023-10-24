from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
#path for the data and vector DB
DATA_PATH = '/home/ubuntu/ProBot/Data'
DB_FAISS_PATH = '/home/ubuntu/ProBot/Data/vectorstore/db_faiss'

load_dotenv()
# Create vector database
def create_vector_db():
    
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,
                                                   chunk_overlap=150)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-distilbert-cos-v1",
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print("Data ingested into the VectorStore")

if __name__ == "__main__":
    create_vector_db()
