import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    
    df = df.drop(columns=['employee_id'])  # not useful

    num_features = df.select_dtypes(include=['int64', 'float64']).columns

    cat_features = df.select_dtypes(include=['object', 'category', 'bool']).columns
    df['tenure_years'] = df['tenure_years'].clip(lower=0).fillna(0)
    df['tenure_years'] = np.log1p(df['tenure_years'])
    from scipy.stats import skew
    df[num_features].apply(skew).sort_values(ascending=False)
    X = df.drop('enrolled', axis=1)
    y = df['enrolled']
    num_features = num_features.drop('enrolled')

    scaler = StandardScaler()

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, num_features),
            ('cat', categorical_transformer, cat_features)
        ]
    )

    X_transformed = preprocessor.fit_transform(X)
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
    all_feature_names = list(num_features) + list(cat_names)

    X = pd.DataFrame(X_transformed, columns=all_feature_names, index=X.index)
    return X, y, preprocessor


def split_and_save_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test