from crewai import Agent
from utils.llm_config import LLMConfig

def create_query_executor_agent():
    llm_config = LLMConfig()
    
    return Agent(
        role="Query Execution Validator",
        goal="Validate and optimize PySpark queries before execution",
        backstory="""You are a database performance expert who validates SQL queries 
        for syntax correctness, performance optimization, and potential issues. 
        You ensure queries are safe to execute and will return meaningful results.""",
        verbose=True,
        allow_delegation=False,
        llm=llm_config.get_llm()
    )