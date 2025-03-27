import os
import chainlit as cl
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Retrieve API Key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Error: GROQ_API_KEY is not set. Please check your .env file.")

# Initialize Groq Chat model
groq_chat = ChatGroq(
    groq_api_key=api_key,
    model_name="llama3-8b-8192"
)

groq_sys_prompt = ChatPromptTemplate.from_template(
    """You are very smart at everything, you always give the best,
    the most accurate and most precise answers. Answer the following Question: {user_prompt}.
    Start the answer directly. No small talk please"""
)

@cl.on_message
async def main(message: cl.Message):
    user_prompt = message.content
    chain = groq_sys_prompt | groq_chat | StrOutputParser()
    response = chain.invoke({"user_prompt": user_prompt})
    
    # Await the asynchronous send operation
    await cl.Message(content=response).send()
