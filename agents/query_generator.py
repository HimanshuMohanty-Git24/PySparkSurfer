from crewai import Agent
from utils.llm_config import LLMConfig

def create_query_generator_agent():
    llm_config = LLMConfig()
    
    return Agent(
        role="PySpark Query Generator",
        goal="Convert natural language requests into accurate PySpark SQL queries",
        backstory="""You are a senior data engineer with expertise in PySpark and SQL. 
        You excel at translating natural language requirements into efficient, 
        syntactically correct PySpark SQL queries. You understand complex data 
        operations, aggregations, filtering, and joins.""",
        verbose=True,
        allow_delegation=False,
        llm=llm_config.get_llm()
    )