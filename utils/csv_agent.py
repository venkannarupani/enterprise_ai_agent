import pandas as pd

def analyze_operational_data(csv_path):

    df = pd.read_csv(csv_path)

    insights = {}

    insights["Total Records"] = len(df)

    insights["Columns"] = list(df.columns)

    insights["Missing Values"] = df.isnull().sum().to_dict()

    numeric_cols = df.select_dtypes(include='number').columns

    insights["Statistics"] = df[numeric_cols].describe().to_dict()

    return insights