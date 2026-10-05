from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma 
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate


embedding_model = MistralAIEmbeddings(model="mistral-embed")

#loaded vector store from create_database
vectorstore = Chroma(
  persist_directory= "chroma_db",
  embedding_function= embedding_model
)

retriever = vectorstore.as_retriever(
  search_type = "mmr",
  search_kwargs = {
    "k" : 4,
    "fetch_k": 10, #similarity type 10 then k(mmr)
    "lambda_mult" : 0.5 #diverce result 0-1
  }
)

llm = ChatMistralAI(model="labs-leanstral-1-5")

#prompt template 
prompt = ChatPromptTemplate.from_messages(
  [
    ("system",
    """
        You are a helpful AI assistan.
        Usy only the provided context to answer the question.
        if the answer is not present in the context,
        say: "I could not find the answer in the document."
    """),
    ("human",
    """
        Context: {context}
        Qustion: {question}

    """)
  ]
)

print("Rag System Created")

print("Press 0 to exit")

while True:
  query = input("You : ")

  if query == "0":
    break

  docs = retriever.invoke(query)

  context = "\n\n".join(
    [doc.page_content for doc in docs]
  )

  final_prompt = prompt.invoke({
    "context": context,
    "question": query
  })

  response = llm.invoke(final_prompt)

  print(f"\n Ai: {response.content}")