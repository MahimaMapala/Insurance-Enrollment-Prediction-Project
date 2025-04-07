

def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, ConfusionMatrixDisplay

    y_pred = model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])


    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


    import pandas as pd

    cm_df = pd.DataFrame(cm, 
                        index=['Actual 0', 'Actual 1'], 
                        columns=['Predicted 0', 'Predicted 1'])
    print(cm_df)



    