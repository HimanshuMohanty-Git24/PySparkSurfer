import streamlit as st
import pandas as pd
import json
import os
from utils.data_processor import DataProcessor
from agents.crew_manager import CrewManager
from config import Config
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import streamlit as st
from pyspark.sql import SparkSession

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="NLP to PySpark Query Generator",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'schema_info' not in st.session_state:
    st.session_state.schema_info = None
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'generated_query' not in st.session_state:
    st.session_state.generated_query = None
if 'spark_session' not in st.session_state:
    st.session_state.spark_session = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def initialize_spark_session():
    """Initialize Spark session if not already created"""
    if st.session_state.spark_session is None:
        try:
            # Stop any existing Spark session
            try:
                from pyspark import SparkContext
                SparkContext.getOrCreate().stop()
            except:
                pass
                
            st.session_state.spark_session = SparkSession.builder \
                .appName("NLP_to_PySpark") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                .getOrCreate()
            
            # Set log level to reduce warnings
            st.session_state.spark_session.sparkContext.setLogLevel("ERROR")
            return True
        except Exception as e:
            st.error(f"Failed to initialize Spark session: {str(e)}")
            return False
    return True

def extract_sql_from_llm_response(llm_response):
    """Extract and clean SQL query from LLM response"""
    response_text = str(llm_response).strip()
    
    # Try to find SQL query in various formats
    sql_patterns = [
        r'```sql\s*(.*?)\s*```',  # SQL code blocks
        r'```\s*(SELECT.*?)\s*```',  # General code blocks with SELECT
        r'(SELECT.*?)(?:\n\n|\Z)',  # SELECT until double newline or end
    ]
    
    import re
    
    for pattern in sql_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if matches:
            sql_query = matches[0].strip()
            break
    else:
        # Fallback: extract manually
        lines = response_text.split('\n')
        sql_lines = []
        capturing = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Start capturing at SELECT
            if line.upper().startswith('SELECT'):
                capturing = True
                sql_lines.append(line)
            elif capturing:
                # Stop at certain markers
                stop_markers = [
                    '"""', '```', 'result_df', 'spark.sql', 'execute the',
                    'to optimize', 'improved_query', 'display the', 'stop the',
                    'spark.stop()', '.show()', '.cache()', '.write.'
                ]
                
                if any(marker in line.lower() for marker in stop_markers):
                    break
                    
                # Stop at Python-like assignments
                if '=' in line and any(keyword in line.lower() for keyword in ['spark', 'df', 'result']):
                    break
                    
                sql_lines.append(line)
        
        sql_query = ' '.join(sql_lines).strip() if sql_lines else ""
    
    # Clean the query
    if sql_query:
        # Remove trailing semicolon
        if sql_query.endswith(';'):
            sql_query = sql_query[:-1]
            
        # Handle placeholder columns by replacing with actual dataset columns
        return sql_query
    
    return None

def clean_sql_query(query):
    """Clean and extract SQL query from potentially messy text"""
    import re
    
    # Remove markdown formatting
    query = query.replace('```sql', '').replace('```', '').strip()
    
    # Remove common prefixes
    query = query.replace('🔍 Generated Query', '').strip()
    
    # Handle the specific case where columns might not exist
    if 'other_column1' in query or 'other_column2' in query:
        # Replace with SELECT * for safety
        query = query.replace('SELECT Item_Outlet_Sales, other_column1, other_column2', 'SELECT *')
    
    # Find the SQL query part
    lines = query.split('\n')
    sql_lines = []
    
    # Look for SELECT statement
    start_capturing = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Start capturing when we see SELECT
        if line.upper().startswith('SELECT'):
            start_capturing = True
            sql_lines.append(line)
        elif start_capturing:
            # Stop if we hit certain keywords that indicate end of query
            if any(keyword in line.upper() for keyword in ['"""', '```', 'RESULT_DF', 'SPARK.SQL', 'TO OPTIMIZE', 'IMPROVED_QUERY', 'EXECUTE THE', 'DISPLAY THE', 'STOP THE']):
                break
            # Stop if line looks like Python code
            if any(keyword in line for keyword in ['=', 'spark.', 'results', '.show()', '.stop()']):
                break
            sql_lines.append(line)
    
    # Join the SQL lines
    clean_query = ' '.join(sql_lines).strip()
    
    # Remove trailing semicolon if present
    if clean_query.endswith(';'):
        clean_query = clean_query[:-1]
    
    return clean_query

def process_llm_response_to_structured_query(llm_response, dataset_schema):
    """Process actual LLM response and extract structured SQL query"""
    response_text = str(llm_response).strip()
    
    # Extract SQL query using multiple approaches
    sql_query = None
    
    # Method 1: Look for SQL code blocks
    import re
    sql_patterns = [
        r'```sql\s*(.*?)\s*```',
        r'```\s*(SELECT.*?)\s*```',
        r'(SELECT.*?)(?=\n\n|\n"""|$)',
    ]
    
    for pattern in sql_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if matches:
            sql_query = matches[0].strip()
            break
    
    # Method 2: Manual extraction if patterns fail
    if not sql_query:
        lines = response_text.split('\n')
        sql_lines = []
        capturing = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Start capturing at SELECT
            if line.upper().startswith('SELECT'):
                capturing = True
                sql_lines.append(line)
            elif capturing:
                # Stop conditions
                stop_conditions = [
                    '"""', '```', 'result_df =', 'spark.sql(',
                    'to optimize', 'improved_query', 'execute the',
                    'spark.stop()', '.show()', '.cache()'
                ]
                
                if any(condition in line.lower() for condition in stop_conditions):
                    break
                    
                # Continue if it looks like part of SQL
                if any(keyword in line.upper() for keyword in ['FROM', 'WHERE', 'ORDER BY', 'GROUP BY', 'HAVING', 'LIMIT', 'JOIN', 'AND', 'OR']):
                    sql_lines.append(line)
                elif line and not any(char in line for char in ['=', '()', 'spark', 'df']):
                    sql_lines.append(line)
                else:
                    break
        
        if sql_lines:
            sql_query = ' '.join(sql_lines).strip()
    
    # Clean and validate the query
    if sql_query:
        # Remove trailing punctuation
        sql_query = sql_query.rstrip(';')
        
        # Validate against schema and fix common issues
        validated_query = validate_and_fix_query(sql_query, list(dataset_schema['columns_info'].keys()))
        
        return {
            'success': True,
            'query': validated_query,
            'original_response': response_text,
            'cleaned_query': sql_query
        }
    
    return {
        'success': False,
        'error': 'Could not extract valid SQL query from LLM response',
        'original_response': response_text
    }

def execute_query_pandas_fallback(query, dataset):
    """Execute query using pandas/sqlite as fallback"""
    try:
        import sqlite3
        import tempfile
        
        # Create temporary SQLite database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            conn = sqlite3.connect(tmp_file.name)
            
            # Write dataset to SQLite
            dataset.to_sql('dataset', conn, index=False, if_exists='replace')
            
            # Execute query
            result_df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Clean up
            os.unlink(tmp_file.name)
            
            return result_df, None
            
    except Exception as e:
        return None, str(e)

def execute_query_and_show_results(query, dataset):
    """Execute the generated query and return results"""
    try:
        # Clean the query thoroughly
        clean_query = clean_sql_query(query)
        
        if not clean_query or not clean_query.upper().startswith('SELECT'):
            return None, "No valid SQL query found in the generated text"
        
        # Ensure dataset is a Pandas DataFrame
        if not isinstance(dataset, pd.DataFrame):
            dataset = pd.DataFrame(dataset)
        
        # Get actual dataset columns
        available_columns = list(dataset.columns)
        
        # Validate and fix query using actual column names
        clean_query = validate_and_fix_query(clean_query, available_columns)
        
        if not clean_query:
            return None, "Could not create valid query for the dataset"
        
        # Try Spark first
        if initialize_spark_session():
            try:
                # Create Spark DataFrame using schema inference
                spark_df = st.session_state.spark_session.createDataFrame(dataset)
                spark_df.createOrReplaceTempView("dataset")
                
                # Execute the query
                result_df = st.session_state.spark_session.sql(clean_query)
                
                # Convert to Pandas for display
                pandas_df = result_df.toPandas()
                
                return pandas_df, None
                
            except Exception as spark_error:
                st.warning(f"Spark execution failed, using fallback method: {str(spark_error)}")
                # Fall back to pandas/sqlite method
                return execute_query_pandas_fallback(clean_query, dataset)
        else:
            # Use pandas/sqlite fallback
            return execute_query_pandas_fallback(clean_query, dataset)
            
    except Exception as e:
        return None, str(e)

def validate_and_fix_query(query, dataset_columns):
    """Validate SQL query against actual dataset columns and fix common issues"""
    if not query:
        return None
        
    import re
    
    # Replace common placeholder columns with actual columns
    placeholder_replacements = {
        'other_column1': dataset_columns[1] if len(dataset_columns) > 1 else dataset_columns[0],
        'other_column2': dataset_columns[2] if len(dataset_columns) > 2 else dataset_columns[0],
        'column1': dataset_columns[0],
        'column2': dataset_columns[1] if len(dataset_columns) > 1 else dataset_columns[0],
    }
    
    for placeholder, actual_col in placeholder_replacements.items():
        query = query.replace(placeholder, actual_col)
    
    # Check if query references non-existent columns
    # Find all column references in SELECT clause
    select_match = re.search(r'SELECT\s+(.+?)\s+FROM', query, re.IGNORECASE)
    if select_match:
        select_part = select_match.group(1)
        
        # Skip if it's SELECT *
        if select_part.strip() != '*':
            # Extract column names (simple parsing)
            columns_in_query = [col.strip() for col in select_part.split(',')]
            
            # Check if any column doesn't exist in dataset
            invalid_columns = []
            for col in columns_in_query:
                # Remove functions, aliases, etc. for basic column name
                basic_col = re.sub(r'[^a-zA-Z0-9_].*', '', col.strip())
                if basic_col and basic_col not in dataset_columns:
                    invalid_columns.append(basic_col)
            
            # If invalid columns found, use SELECT *
            if invalid_columns:
                query = re.sub(r'SELECT\s+.+?\s+FROM', 'SELECT * FROM', query, flags=re.IGNORECASE)
    
    return query

def main():
    st.title("🔍 Natural Language to PySpark Query Generator")
    st.markdown("Upload your dataset and ask questions in natural language!")
    
    # Sidebar for dataset upload
    with st.sidebar:
        st.header("📁 Dataset Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
            help="Supported formats: CSV, Excel, JSON, Parquet"
        )
        
        if uploaded_file is not None:
            if allowed_file(uploaded_file.name):
                # Save uploaded file
                file_path = os.path.join(Config.UPLOAD_FOLDER, uploaded_file.name)
                os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load dataset
                try:
                    file_type = uploaded_file.name.rsplit('.', 1)[1].lower()
                    df = st.session_state.data_processor.load_dataset(file_path, file_type)
                    st.session_state.dataset = df
                    st.session_state.schema_info = st.session_state.data_processor.get_schema_info(df)
                    
                    st.success("✅ Dataset loaded successfully!")
                    
                    # Display dataset info
                    st.subheader("📊 Dataset Information")
                    st.write(f"**Rows:** {st.session_state.schema_info['total_rows']:,}")
                    st.write(f"**Columns:** {st.session_state.schema_info['total_columns']}")
                    
                    # Show column information
                    st.subheader("📋 Column Details")
                    columns_df = pd.DataFrame(st.session_state.schema_info['columns'])
                    st.dataframe(columns_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error loading dataset: {str(e)}")
            else:
                st.error("❌ Unsupported file type")
    
    # Main content area
    if st.session_state.dataset is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("💬 Ask Your Question")
            
            # Sample questions
            st.subheader("💡 Sample Questions")
            sample_questions = [
                "Show me the top 10 records",
                "Find the highest sales by category",
                "What is the average price for each product type?",
                "Show monthly sales trends",
                "Which items have sales greater than 1000?"
            ]
            
            selected_sample = st.selectbox(
                "Choose a sample question or write your own:",
                [""] + sample_questions
            )
            
            # Natural language input
            natural_language_query = st.text_area(
                "Enter your question in natural language:",
                value=selected_sample,
                height=100,
                placeholder="e.g., Find the highest sales of last month"
            )
            
            col_btn1, col_btn2 = st.columns([1, 1])
            
            with col_btn1:
                generate_query = st.button("🚀 Generate Query", type="primary")
            
            with col_btn2:
                if st.button("🔄 Clear"):
                    st.rerun()
        
        with col2:
            st.header("📈 Dataset Preview")
            if st.session_state.schema_info:
                preview_df = pd.DataFrame(st.session_state.schema_info['sample_data'])
                st.dataframe(preview_df, use_container_width=True)
        
        # Process query
        if generate_query and natural_language_query.strip():
            with st.spinner("🤖 AI agents are working on your query..."):
                try:
                    crew_manager = CrewManager()
                    
                    # Generate query using CrewAI
                    query_result = crew_manager.process_natural_language_query(
                        natural_language_query, 
                        st.session_state.schema_info
                    )
                    
                    # Process the actual LLM response to extract structured query
                    structured_result = process_llm_response_to_structured_query(
                        query_result, 
                        st.session_state.schema_info
                    )
                    
                    if structured_result['success']:
                        # Store cleaned query in session state
                        st.session_state.generated_query = structured_result['query']
                        
                        # Show debug info if needed
                        if st.checkbox("Show debug info", value=False):
                            st.expander("Debug Information").write({
                                "Original Response": structured_result['original_response'],
                                "Cleaned Query": structured_result['cleaned_query'],
                                "Final Query": structured_result['query']
                            })
                    else:
                        st.error(f"❌ {structured_result['error']}")
                        
                        # Show the original response for debugging
                        st.expander("Original LLM Response").write(structured_result['original_response'])
                    
                except Exception as e:
                    st.error(f"❌ Error generating query: {str(e)}")
        
        elif generate_query and not natural_language_query.strip():
            st.warning("⚠️ Please enter a question first!")
        
        # Display generated query if available
        if st.session_state.generated_query:
            st.header("🔍 Generated Query")
            st.code(st.session_state.generated_query, language='sql')
            
            # Execute query button
            if st.button("⚡ Execute Query", type="secondary"):
                # Initialize Spark session
                initialize_spark_session()
                
                st.subheader("📊 Query Results")
                
                with st.spinner("Executing query..."):
                    results_df, error = execute_query_and_show_results(
                        st.session_state.generated_query, 
                        st.session_state.dataset
                    )
                    
                if error:
                    st.error(f"❌ Error executing query: {error}")
                elif results_df is not None:
                    if len(results_df) > 0:
                        # Display metrics
                        col_metric1, col_metric2, col_metric3 = st.columns(3)
                        with col_metric1:
                            st.metric("Rows Returned", len(results_df))
                        with col_metric2:
                            st.metric("Columns", len(results_df.columns))
                        with col_metric3:
                            st.metric("Query Status", "✅ Success")
                        
                        # Display results table
                        st.subheader("📋 Results Table")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Auto-generate visualizations if applicable
                        if len(results_df.columns) >= 2 and len(results_df) > 1:
                            st.subheader("📈 Visualization")
                            
                            numeric_cols = results_df.select_dtypes(include=['number']).columns
                            categorical_cols = results_df.select_dtypes(include=['object']).columns
                            
                            if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                                # Create bar chart
                                fig = px.bar(
                                    results_df.head(20), 
                                    x=categorical_cols[0], 
                                    y=numeric_cols[0],
                                    title=f"{numeric_cols[0]} by {categorical_cols[0]}"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            elif len(numeric_cols) >= 2:
                                # Create scatter plot
                                fig = px.scatter(
                                    results_df.head(100), 
                                    x=numeric_cols[0], 
                                    y=numeric_cols[1],
                                    title=f"{numeric_cols[1]} vs {numeric_cols[0]}"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="query_results.csv",
                            mime="text/csv"
                        )
                        
                        st.success(f"Query executed successfully! Found {len(results_df)} records.")
                    else:
                        st.info("Query executed successfully but returned no results.")
                else:
                    st.error("❌ Unknown error occurred during query execution.")
    
    else:
        st.info("👆 Please upload a dataset to get started!")
        
        # Show example datasets
        st.header("📚 Example Use Cases")
        
        examples = [
            {
                "title": "Sales Analysis",
                "description": "Analyze sales data by product, region, and time period",
                "queries": [
                    "Show top 10 products by sales",
                    "Find monthly sales trends",
                    "Which region has highest revenue?"
                ]
            },
            {
                "title": "Customer Analytics",
                "description": "Understand customer behavior and preferences",
                "queries": [
                    "Show customer segments by age",
                    "Find repeat customers",
                    "What are the most popular products?"
                ]
            },
            {
                "title": "Financial Analysis",
                "description": "Analyze revenue, costs, and profitability",
                "queries": [
                    "Calculate profit margins by category",
                    "Show quarterly revenue growth",
                    "Find cost centers with highest expenses"
                ]
            }
        ]
        
        cols = st.columns(len(examples))
        for i, example in enumerate(examples):
            with cols[i]:
                st.subheader(example["title"])
                st.write(example["description"])
                for query in example["queries"]:
                    st.code(f'"{query}"', language='text')

if __name__ == "__main__":
    main()