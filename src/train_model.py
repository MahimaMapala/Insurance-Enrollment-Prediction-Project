

def train_and_save_model(X_train, y_train):
    from sklearn.linear_model import LogisticRegression

    # using this model because we have smaller data set
    # and works well for binary classification problems
    # also, it is fast to train and predict

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)


    return model

