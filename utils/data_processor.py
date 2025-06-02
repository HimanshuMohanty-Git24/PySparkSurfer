import pandas as pd
import json
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from config import Config

class DataProcessor:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName(Config.SPARK_CONFIG['spark.app.name']) \
            .master(Config.SPARK_CONFIG['spark.master']) \
            .config("spark.executor.memory", Config.SPARK_CONFIG['spark.executor.memory']) \
            .config("spark.driver.memory", Config.SPARK_CONFIG['spark.driver.memory']) \
            .getOrCreate()
    
    def load_dataset(self, file_path, file_type):
        """Load dataset and return DataFrame with schema info"""
        try:
            if file_type == 'csv':
                df = self.spark.read.csv(file_path, header=True, inferSchema=True)
            elif file_type in ['xlsx', 'xls']:
                # Convert Excel to CSV first using pandas
                pandas_df = pd.read_excel(file_path)
                temp_csv = file_path.replace(f'.{file_type}', '_temp.csv')
                pandas_df.to_csv(temp_csv, index=False)
                df = self.spark.read.csv(temp_csv, header=True, inferSchema=True)
            elif file_type == 'json':
                df = self.spark.read.json(file_path)
            elif file_type == 'parquet':
                df = self.spark.read.parquet(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            return df
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def get_schema_info(self, df):
        """Extract schema information from DataFrame"""
        schema_info = {
            'columns': [],
            'total_rows': df.count(),
            'total_columns': len(df.columns),
            'sample_data': df.limit(5).toPandas().to_dict('records')
        }
        
        for field in df.schema.fields:
            schema_info['columns'].append({
                'name': field.name,
                'type': str(field.dataType),
                'nullable': field.nullable
            })
        
        return schema_info
    
    def execute_query(self, df, query):
        """Execute PySpark query on DataFrame"""
        try:
            # Create temporary view
            df.createOrReplaceTempView("dataset")
            
            # Execute query
            result_df = self.spark.sql(query)
            
            # Convert to pandas for display
            result_pandas = result_df.toPandas()
            
            return {
                'success': True,
                'data': result_pandas.to_dict('records'),
                'columns': result_pandas.columns.tolist(),
                'row_count': len(result_pandas)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def close_spark(self):
        """Close Spark session"""
        self.spark.stop()