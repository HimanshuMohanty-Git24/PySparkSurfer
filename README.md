# 🔍 PySparkSurfer: Natural Language to PySpark Query Generator

**PySparkSurfer** is an interactive Streamlit-based web app that enables users to generate and run PySpark SQL queries by simply asking questions in natural language. Upload your dataset, type a question like *"Find the average sales per region"* and get back executable PySpark SQL, results, and visualizations — all with the help of LLM-powered CrewAI agents.

---

## 🧠 How It Works

1. **Upload Dataset**: Upload your `.csv`, `.xlsx`, `.json`, or `.parquet` dataset.
2. **Ask Questions**: Type queries like:

   * "Show top 10 products by sales"
   * "Which category has highest revenue?"
3. **LLM Agents in Action**:

   * `Schema Analyzer`: Understands dataset structure.
   * `Query Generator`: Converts questions into PySpark SQL.
   * `Query Validator`: Optimizes and validates the query.
4. **Execute & Visualize**: Run the query using PySpark or fallback to SQLite, then visualize results.

---

## 📁 Project Structure

```
himanshumohanty-git24-pysparksurfer/
├── app.py                      # Streamlit app entry point
├── config.py                  # Configuration and environment variables
├── agents/                    # CrewAI agents
│   ├── __init__.py
│   ├── crew_manager.py        # Orchestrates schema analysis and query generation
│   ├── query_executor.py      # Validates and optimizes SQL queries
│   ├── query_generator.py     # Translates natural language to SQL
│   └── schema_analyzer.py     # Analyzes dataset schema
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── data_processor.py      # Handles file parsing and Spark integration
│   └── llm_config.py          # LLM configuration and integration
└── README.md
```

---

## 🚀 Features

✅ Upload datasets in multiple formats
✅ Ask data questions in plain English
✅ Get auto-generated, validated PySpark SQL queries
✅ Preview datasets and results
✅ Visualize query output using Plotly
✅ Download results as CSV

---

## 🧱 Built With

* [Streamlit](https://streamlit.io/) – Frontend UI
* [PySpark](https://spark.apache.org/docs/latest/api/python/) – Backend query execution
* [CrewAI](https://docs.crewai.com/) – Multi-agent orchestration
* [GROQ](https://groq.com/) – LLM for natural language understanding
* [Plotly](https://plotly.com/python/) – Data visualization
* [SQLite (fallback)](https://www.sqlite.org/index.html) – Query execution fallback for small data

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/himanshumohanty/himanshumohanty-git24-pysparksurfer.git
cd himanshumohanty-git24-pysparksurfer
```

### 2. Install Dependencies

Make sure Python 3.8+ is installed.

```bash
pip install -r requirements.txt
```

> **Note**: You must also have Java installed for PySpark to work correctly.

### 3. Set Environment Variables

Create a `.env` file and add your API key:

```
GROQ_API_KEY=your_groq_api_key
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

---

## 💡 Example Questions

* "Show the top 5 products with the highest sales"
* "Find average revenue per store"
* "Which customers made purchases above 1000?"
* "Monthly sales trend by region"
* "Total number of transactions per category"

---

## 🧠 Architecture

```
[ User ]
   ↓
[Natural Language Query]
   ↓
[CrewManager (Agents Pipeline)]
   ↓
[Query Generation & Validation]
   ↓
[PySpark Execution / SQLite Fallback]
   ↓
[Data Visualization + Download]
```

---

## 🛠️ Configuration

All config options (upload folder, Spark memory, allowed file types) are managed in `config.py`.

---

## 📦 Dependencies

Add a `requirements.txt` like:

```
streamlit
pandas
openpyxl
plotly
pyspark
python-dotenv
crewai
groq
```

---

## 🧪 Test Datasets

You can try with open datasets like:

* [Kaggle Retail Sales Dataset](https://www.kaggle.com/datasets)
* [Public CSV Samples](https://people.sc.fsu.edu/~jburkardt/data/csv/)

---

## 🧑‍💻 Author

**Himanshu Mohanty**
Passionate about AI, NLP, and LLM-based intelligent systems.
[GitHub](https://github.com/himanshumohanty) • [LinkedIn](https://www.linkedin.com/in/himanshumohanty/)

---

## 📄 License

This project is open-sourced under the [MIT License](LICENSE).
