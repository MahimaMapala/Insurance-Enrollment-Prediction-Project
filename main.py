from src.data_preprocessing import load_data, preprocess_data, split_and_save_data
from src.train_model import train_and_save_model
from src.evaluate_model import evaluate_model

import pandas as pd
import numpy as np


def main():
    df=pd.read_csv('structured_code/data/employee_data.csv')
    X, y, preprocessor = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_and_save_data(X, y)

    model=train_and_save_model(X_train, y_train)

    import joblib
    joblib.dump(model, 'structured_code/models/model.pkl')
    joblib.dump(preprocessor, 'structured_code/models/preprocessor.pkl')

    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
