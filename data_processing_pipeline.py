import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
    
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            logging.info("Data loaded successfully from {}".format(self.file_path))
        except Exception as e:
            logging.error("Error loading data: {}".format(e))

    def data_overview(self):
        if self.df is not None:
            logging.info("Data Overview:")
            logging.info("Shape of the dataset: {}".format(self.df.shape))
            logging.info("Columns: {}".format(self.df.columns.tolist()))
            logging.info("First 5 rows:\n{}".format(self.df.head()))
        else:
            logging.warning("DataFrame is empty. Load data first.")

    def handle_missing_values(self):
        if self.df is not None:
            missing_values = self.df.isnull().sum()
            missing_columns = missing_values[missing_values > 0].index.tolist()
            logging.info("Missing values by column:\n{}".format(missing_values[missing_values > 0]))

            imputer = SimpleImputer(strategy='mean')
            for col in missing_columns:
                if self.df[col].dtype in ['float64', 'int64']:
                    self.df[col] = imputer.fit_transform(self.df[[col]])
                    logging.info("Missing values filled for column: {}".format(col))
            logging.info("Missing values handled.")
        else:
            logging.warning("DataFrame is empty. Load data first.")

    def remove_duplicates(self):
        if self.df is not None:
            initial_shape = self.df.shape
            self.df.drop_duplicates(inplace=True)
            logging.info("Removed duplicates: {} to {}".format(initial_shape, self.df.shape))
        else:
            logging.warning("DataFrame is empty. Load data first.")

    def transform_features(self):
        if self.df is not None:
            # Example transformation: create interaction features
            for col1 in self.df.select_dtypes(include=['float64', 'int64']).columns:
                for col2 in self.df.select_dtypes(include=['float64', 'int64']).columns:
                    if col1 != col2:
                        new_col_name = "{}_x_{}".format(col1, col2)
                        self.df[new_col_name] = self.df[col1] * self.df[col2]
                        logging.info("Created new feature: {}".format(new_col_name))
        else:
            logging.warning("DataFrame is empty. Load data first.")

    def standardize_features(self):
        if self.df is not None:
            scaler = StandardScaler()
            numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
            self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
            logging.info("Standardized features:\n{}".format(numeric_cols.tolist()))
        else:
            logging.warning("DataFrame is empty. Load data first.")

    def visualize_data(self):
        if self.df is not None:
            sns.pairplot(self.df)
            plt.title("Pairplot of the Dataset")
            plt.savefig("pairplot.png")
            plt.close()
            logging.info("Saved pairplot to pairplot.png")

    def save_processed_data(self, output_path: str):
        if self.df is not None:
            self.df.to_csv(output_path, index=False)
            logging.info("Processed data saved to {}".format(output_path))
        else:
            logging.warning("DataFrame is empty. Load data first.")

if __name__ == "__main__":
    processor = DataProcessor("input_data.csv")
    processor.load_data()
    processor.data_overview()
    processor.handle_missing_values()
    processor.remove_duplicates()
    processor.transform_features()
    processor.standardize_features()
    processor.visualize_data()
    processor.save_processed_data("processed_data.csv")