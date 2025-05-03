import pandas as pd
import numpy as np
import uuid
import matplotlib.pyplot as plt
from astrapy import DataAPIClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class MedallionPipeline:
    def __init__(self, db_config, input_file):
        self.db_config = db_config
        self.input_file = input_file
        self.client = DataAPIClient(db_config["ASTRA_DB_APPLICATION_TOKEN"])
        self.db = self.client.get_database_by_api_endpoint(
            db_config["ASTRA_DB_API_ENDPOINT"], keyspace=db_config["KEYSPACE_NAME"]
        )

    def ensure_collection_exists(self, collection_name):
        try:
            if collection_name not in self.db.list_collections():
                self.db.create_collection(collection_name)
        except Exception as e:
            print(f"Error ensuring collection {collection_name} exists: {e}")

    def load_raw_data(self):
        df = pd.read_csv(self.input_file)
        # Show number of rows and columns
        num_rows, num_columns = df.shape
        print(f"Dataset Dimensions: {num_rows} rows × {num_columns} columns")
        self.ensure_collection_exists("bronze_sales")
        collection = self.db.get_collection("bronze_sales")
        for _, row in df.iterrows():
            doc = row.to_dict()
            doc["_id"] = str(uuid.uuid4())
            collection.insert_one(doc)
        print("Raw data loaded into Bronze table.")

    def clean_data(self):
        collection = self.db.get_collection("bronze_sales")
        data = collection.find()
        df = pd.DataFrame(data)

        df.drop_duplicates(subset=["Transaction_ID"], keep="first", inplace=True)
        df.fillna({"Purchase_Frequency": 0, "Lifetime_Value": 0.0, "Churn_Probability": 0.0}, inplace=True)
        df["Launch_Date"] = pd.to_datetime(df["Launch_Date"], errors="coerce", utc=True)

        self.ensure_collection_exists("silver_sales")
        silver_collection = self.db.get_collection("silver_sales")
        for _, row in df.iterrows():
            doc = row.to_dict()
            doc["_id"] = str(uuid.uuid4())
            silver_collection.insert_one(doc)
        print("Data cleaned and stored in Silver table.")

    def aggregate_data(self):
        collection = self.db.get_collection("silver_sales")
        data = collection.find()
        df = pd.DataFrame(data)

        self.ensure_collection_exists("gold_sales_by_region")
        self.ensure_collection_exists("gold_sales_by_category")

        gold_sales_by_region = df.groupby("Region")["Lifetime_Value"].sum().reset_index()
        collection_region = self.db.get_collection("gold_sales_by_region")
        for _, row in gold_sales_by_region.iterrows():
            doc = row.to_dict()
            doc["_id"] = str(uuid.uuid4())
            collection_region.insert_one(doc)

        gold_sales_by_category = df.groupby("Most_Frequent_Category")["Churn_Probability"].mean().reset_index()
        collection_category = self.db.get_collection("gold_sales_by_category")
        for _, row in gold_sales_by_category.iterrows():
            doc = row.to_dict()
            doc["_id"] = str(uuid.uuid4())
            collection_category.insert_one(doc)

        print("Aggregated insights stored in Gold tables.")

    def generate_visualizations(self):
        collection = self.db.get_collection("gold_sales_by_region")
        data = collection.find()
        df = pd.DataFrame(data)

        plt.figure(figsize=(10, 5))
        plt.bar(df["Region"], df["Lifetime_Value"])
        plt.title("Lifetime Value by Region")
        plt.xlabel("Region")
        plt.ylabel("Lifetime Value")
        plt.xticks(rotation=45)
        plt.show()

        collection = self.db.get_collection("gold_sales_by_category")
        data = collection.find()
        df = pd.DataFrame(data)

        plt.figure(figsize=(10, 5))
        plt.bar(df["Most_Frequent_Category"], df["Churn_Probability"])
        plt.title("Churn Probability by Category")
        plt.xlabel("Category")
        plt.ylabel("Churn Probability")
        plt.xticks(rotation=45)
        plt.show()
        print("Visualizations generated.")

    def train_ml_model(self):
        collection = self.db.get_collection("silver_sales")
        data = collection.find()
        df = pd.DataFrame(data)

        X = df[["Purchase_Frequency", "Average_Order_Value", "Time_Between_Purchases", "Lifetime_Value"]]
        y = (df["Churn_Probability"] > 0.5).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"ML Model Accuracy: {accuracy:.2f}")

    def run_pipeline(self):
        self.load_raw_data()
        self.clean_data()
        self.aggregate_data()
        self.generate_visualizations()
        self.train_ml_model()
        print("Medallion Architecture pipeline completed.")

# Database Configuration
db_config = {
    "ASTRA_DB_APPLICATION_TOKEN": os.getenv(ASTRA_DB_APPLICATION_TOKEN),
    "ASTRA_DB_API_ENDPOINT": os.getenv(ASTRA_DB_API_ENDPOINT),
    "KEYSPACE_NAME": os.getenv(KEYSPACE_NAME)
}
DATASET_URL = "https://raw.githubusercontent.com/Dhinesh2454/Sales-and-Customer-Insights-Dataset/main/sales_and_customer_insights.csv"

# Run the pipeline
pipeline = MedallionPipeline(db_config=db_config, input_file=DATASET_URL)
pipeline.run_pipeline()
