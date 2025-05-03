# 🌿 InsightBloom: Sales Intelligence with Cassandra + ML

InsightBloom is a complete Medallion Architecture pipeline built using Python, Astra DB (Cassandra-as-a-Service), and machine learning. It ingests raw sales/customer data, cleans it, stores it in Astra DB, and creates Gold-level insights and ML predictions.

---

## 🏗️ Architecture Overview

```
Raw CSV
  │
  ▼
Bronze Layer → stores raw ingested records
  │
  ▼
Silver Layer → cleansed, typed, deduplicated data
  │
  ▼
Gold Layer → aggregations:
             - Lifetime Value by Region
             - Churn Probability by Category
             - Visuals & ML (RandomForest)
```

---

## ⚙️ Prerequisites

- Python 3.7 or higher
- Git
- DataStax Astra DB account (free)

---

## 🚀 Astra DB Setup

1. Sign up at https://www.datastax.com/astra
2. Create a database (e.g. sales_db)
3. Go to “Connect → Data API” and copy:
   - API Endpoint
   - Application Token (use Admin role)
4. In MedallionPipeline.py, replace the following fields in db_config:

```python
db_config = {
    "ASTRA_DB_APPLICATION_TOKEN": "<your_token_here>",
    "ASTRA_DB_API_ENDPOINT": "<your_endpoint_here>",
    "KEYSPACE_NAME": "<your_keyspace_name>"
}
```

---

## 📦 Installation

Clone the project and install dependencies:

```bash
git clone https://github.com/LokaPoojithaDondeti/InsightBloom-Sales-Intelligence-with-Cassandra-ML.git
cd InsightBloom-Sales-Intelligence-with-Cassandra-ML

# Optional: Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

📝 Create a .env file or directly modify db_config in the script with your Astra DB credentials.

---

## 📊 Run the Pipeline

Run the end-to-end Medallion pipeline with:

```bash
python MedallionPipeline.py
```

You’ll see:

- Dataset shape
- Data loaded → Bronze
- Cleaned & saved → Silver
- Aggregated insights → Gold
- 2 bar charts
- ML model trained with accuracy score

---

## 🧪 Machine Learning

- Algorithm: RandomForestClassifier
- Target: Churn Probability (binarized)
- Features: Purchase Frequency, Order Value, Time Between Purchases, Lifetime Value
- Output: Accuracy score printed

---

## 📈 Visualizations

1. 📊 Lifetime Value by Region
2. 📊 Avg. Churn Probability by Category

Generated using Matplotlib and displayed at runtime.

---
