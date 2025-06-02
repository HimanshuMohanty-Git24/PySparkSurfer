from crewai import Crew, Task
from .schema_analyzer import create_schema_analyzer_agent
from .query_generator import create_query_generator_agent
from .query_executor import create_query_executor_agent

class CrewManager:
    def __init__(self):
        self.schema_analyzer = create_schema_analyzer_agent()
        self.query_generator = create_query_generator_agent()
        self.query_executor = create_query_executor_agent()
    
    def process_natural_language_query(self, natural_language, schema_info):
        """Process natural language query using CrewAI agents"""
        
        # Task 1: Analyze schema and understand context
        schema_analysis_task = Task(
            description=f"""
            Analyze the following dataset schema and provide insights:
            
            Schema Information:
            {schema_info}
            
            Natural Language Query: {natural_language}
            
            Provide a detailed analysis of:
            1. Relevant columns for the query
            2. Data types and their implications
            3. Potential relationships between columns
            4. Any data quality considerations
            """,
            agent=self.schema_analyzer,
            expected_output="Detailed schema analysis with relevant columns and relationships"
        )
        
        # Task 2: Generate PySpark query
        query_generation_task = Task(
            description=f"""
            Based on the schema analysis and natural language query, generate a PySpark SQL query.
            
            Natural Language Query: {natural_language}
            Schema Info: {schema_info}
            
            Requirements:
            1. Generate syntactically correct PySpark SQL
            2. Use appropriate aggregations and filters
            3. Handle date/time operations if needed
            4. Ensure the query addresses the user's request accurately
            5. Use the table name 'dataset' in your query
            
            Return only the SQL query without additional explanation.
            """,
            agent=self.query_generator,
            expected_output="Valid PySpark SQL query",
            context=[schema_analysis_task]
        )
        
        # Task 3: Validate and optimize query
        query_validation_task = Task(
            description=f"""
            Validate the generated PySpark query for:
            1. Syntax correctness
            2. Performance optimization opportunities
            3. Potential runtime errors
            4. Result accuracy
            
            If issues are found, provide an improved version.
            Return the final validated query.
            """,
            agent=self.query_executor,
            expected_output="Validated and optimized PySpark SQL query",
            context=[query_generation_task]
        )
        
        # Create and run crew
        crew = Crew(
            agents=[self.schema_analyzer, self.query_generator, self.query_executor],
            tasks=[schema_analysis_task, query_generation_task, query_validation_task],
            verbose=True
        )
        
        result = crew.kickoff()
        return result