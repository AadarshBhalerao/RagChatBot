from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

pdf_path = 'Rich-Dad-Poor-Dad.pdf'

# create loader and split document
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

# embedding function
embedding_func = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# create vector store
vectordb = Chroma.from_documents(
    documents=pages,
    embedding=embedding_func,
    persist_directory="vector_db",
    collection_name="rich_dad_poor_dad")

# make vector store persistant
vectordb.persist()