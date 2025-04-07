==========================================
Insurance Enrollment Prediction Project
==========================================

This project builds a machine learning pipeline to predict whether an employee will enroll in a new insurance plan based on demographic and employment data.

------------------------------------------
📁 Project Structure
------------------------------------------
- data/
    - employee_data.csv            # Synthetic employee dataset
- models/
    - model.pkl                    # Trained ML model
    - preprocessor.pkl             # Saved preprocessor
- src/
    - data_preprocessing.py       # Data loading and preprocessing
    - train_model.py              # Model training and saving
    - evaluate_model.py           # Model evaluation
- api/
    - app.py                      # FastAPI app to serve predictions
- requirements.txt                # Python package dependencies
- README.txt                      # Setup & usage instructions
- main.py

------------------------------------------
💻 Setup Instructions
------------------------------------------

1️⃣ Create a virtual environment (using `venv`):
------------------------------------------------
python -m venv venv

2️⃣ Activate the virtual environment:
-------------------------------------
venv\Scripts\activate


3️⃣ Install dependencies from requirements.txt:
------------------------------------------------
pip install -r requirements.txt


------------------------------------------
Running the Application
------------------------------------------
python -m uvicorn api.app:app --reload


Once the server is running, open your browser or use tools like Postman to access:

- Docs: http://127.0.0.1:8000/docs
- Root: http://127.0.0.1:8000/

------------------------------------------
📦 Example JSON Payload (for prediction)
------------------------------------------

POST to `/predict/` with:
```json
{
  "age": 42,
  "gender": "Male",
  "marital_status": "Married",
  "salary": 66274.58,
  "employment_type": "Contract",
  "region": "West",
  "has_dependents": "Yes",
  "tenure_years": 1.0
}
```

You will receive:
```json
{
  "enrollment_prediction": 1,
  "probability": 0.78
}
```

------------------------------------------
🛠️ Notes
------------------------------------------
- Be sure the model and preprocessor (`.pkl` files) are in place before using the API.
- You can retrain the model by executing the main script:
```bash
python main.py
```
