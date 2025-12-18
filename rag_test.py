from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7
)
prompt=PromptTemplate(
    input_variables=['animal_type'],
    template="I have a {animal_type}. Suggest me 5 cool names for it."
)

chain = prompt | llm

response = chain.invoke({"animal_type": "Cat"})
print(response.content)


# response = llm.invoke("I have a pet dog. Suggest my 5 cool names for him.")
# print(response.content)
