from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma 

# document loader 
data = PyPDFLoader("document loaders/deeplearning.pdf")
docs = data.load()

# split into chunks 
splitter = RecursiveCharacterTextSplitter(
  chunk_size = 1000,
  chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

#create the embedding
embedding_model = MistralAIEmbeddings(model="mistral-embed")

# store embedding in vectorestore - chroma db
vectorstore = Chroma.from_documents(
  documents=chunks,
  embedding=embedding_model,
  persist_directory="chroma_db"
)