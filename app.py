import streamlit as st
import pandas as pd
import json
import os
from utils.data_processor import DataProcessor
from agents.crew_manager import CrewManager
from config import Config
import plotly.express as px
import plotly.graph_objects as go

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

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

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
                    
                    # Extract the final query from the result
                    generated_query = str(query_result).strip()
                    
                    # Clean up the query (remove any extra text)
                    if "SELECT" in generated_query.upper():
                        lines = generated_query.split('\n')
                        query_lines = []
                        capture = False
                        for line in lines:
                            if "SELECT" in line.upper():
                                capture = True
                            if capture:
                                query_lines.append(line)
                                if line.strip().endswith(';'):
                                    break
                        generated_query = '\n'.join(query_lines).strip()
                    
                    st.header("🔍 Generated Query")
                    st.code(generated_query, language='sql')
                    
                    # Execute query
                    col_exec, col_copy = st.columns([1, 1])
                    
                    with col_exec:
                        execute_query = st.button("⚡ Execute Query", type="secondary")
                    
                    with col_copy:
                        st.code(generated_query, language='sql')
                    
                    if execute_query:
                        with st.spinner("Executing query..."):
                            result = st.session_state.data_processor.execute_query(
                                st.session_state.dataset, 
                                generated_query
                            )
                            
                            if result['success']:
                                st.header("📊 Query Results")
                                
                                if result['data']:
                                    results_df = pd.DataFrame(result['data'])
                                    
                                    # Display metrics
                                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                                    with col_metric1:
                                        st.metric("Rows Returned", result['row_count'])
                                    with col_metric2:
                                        st.metric("Columns", len(result['columns']))
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
                                else:
                                    st.info("Query executed successfully but returned no results.")
                            else:
                                st.error(f"❌ Query execution failed: {result['error']}")
                
                except Exception as e:
                    st.error(f"❌ Error generating query: {str(e)}")
        
        elif generate_query and not natural_language_query.strip():
            st.warning("⚠️ Please enter a question first!")
    
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