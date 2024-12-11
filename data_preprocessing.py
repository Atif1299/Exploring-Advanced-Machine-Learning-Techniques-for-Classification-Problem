import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    if data.isnull().sum().any():
        data.fillna(data.median(numeric_only=True), inplace=True)
    return data