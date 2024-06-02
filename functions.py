import os, openai
from configparser import ConfigParser
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

config = ConfigParser()

try:
  config.read('config.ini')
except:
  print('config.ini format error')
  raise SystemExit()

openai.api_type = config['OpenAI']['api_type']
openai.api_key = config['OpenAI']['api_key']
openai.api_base = config['OpenAI']['api_base']
openai.api_version = config['OpenAI']['api_version']

embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="vector_db",
    collection_name="rich_dad_poor_dad",
    embedding_function=embedding_function,
)

prompt = PromptTemplate(
    template="""Use the following pieces of context to answer the user question.
    chat_history: {chat_history}
    Context: {text}
    Question: {question}
    Answer:""",
    input_variables=["text", "question", "chat_history"]
)

# create chat model
llm = AzureChatOpenAI(
        openai_api_version=openai.api_version,
        openai_api_base=openai.api_base,
        openai_api_key=openai.api_key,
        azure_deployment=config['OpenAI']['model_name']
)

# create memory
memory = ConversationBufferMemory(
    return_messages=True, memory_key="chat_history")

# create retriever chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    retriever=vector_db.as_retriever(
        search_kwargs={'fetch_k': 20, 'k': 4}, search_type='mmr'),
    chain_type="refine",
)

def rag(question: str) -> str:
    # call QA chain
    response = qa_chain({"question": question})

    return response.get("answer")

# question = 'What is the book about'
# answer = qa_chain({
#   "question": question
# })
# print(answer)