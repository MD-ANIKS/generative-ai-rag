from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate



template = ChatPromptTemplate.from_messages(
  [
    ("system", "you are a AI that summarizes the text"),
    ("human", "{data}")
  ]
)

model = ChatMistralAI(model="labs-leanstral-1-5")

