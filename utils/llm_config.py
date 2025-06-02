from groq import Groq
from langchain_groq import ChatGroq
from config import Config

class LLMConfig:
    def __init__(self):
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name="llama3-70b-8192",
            temperature=0.1,
            max_tokens=4000
        )
    
    def get_llm(self):
        return self.llm
    
    def get_client(self):
        return self.client