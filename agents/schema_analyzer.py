from crewai import Agent
from utils.llm_config import LLMConfig

def create_schema_analyzer_agent():
    llm_config = LLMConfig()
    
    return Agent(
        role="Data Schema Analyzer",
        goal="Analyze dataset schema and understand data structure for query generation",
        backstory="""You are an expert data analyst specializing in understanding 
        dataset structures. You analyze column names, data types, and relationships 
        to provide context for query generation.""",
        verbose=True,
        allow_delegation=False,
        llm=llm_config.get_llm()
    )