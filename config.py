import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'csv', 'json', 'xlsx', 'xls', 'parquet'}
    
    # Spark Configuration
    SPARK_CONFIG = {
        'spark.app.name': 'NLP-to-PySpark',
        'spark.master': 'local[*]',
        'spark.executor.memory': '2g',
        'spark.driver.memory': '2g'
    }

# Environment validation
if not Config.GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")