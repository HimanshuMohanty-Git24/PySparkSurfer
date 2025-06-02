import os
from crewai import LLM

class LLMConfig:
    def __init__(self):
        # Set up Groq API key
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        # Initialize the LLM instance using CrewAI LLM class with proper Groq format
        self.llm = LLM(
            model="groq/llama-3.3-70b-versatile",
            temperature=0.1,
            max_completion_tokens=4096
        )
    
    def get_llm(self):
        """Get configured LLM for CrewAI agents"""
        return self.llm
    
    def get_client(self):
        """Return the LLM instance (same as get_llm for compatibility)"""
        return self.llm