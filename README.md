# Customer Churn Prediction & Business Analytics System

Beginner-friendly project using Python, pandas, scikit-learn, Streamlit, and SQLAlchemy. It uses SQLite by default and can use PostgreSQL with `DATABASE_URL`.

## Run on Windows
1. Extract the ZIP and open this folder in VS Code.
2. Open Terminal in VS Code.
3. Run:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. The terminal will show a local URL, usually `http://localhost:8501`.

## Features
- Sample customer churn CSV (synthetic demo data)
- Data quality checks and EDA dashboard
- Train Logistic Regression, Decision Tree, and Random Forest
- Evaluate accuracy, precision, recall, and F1-score
- Enter customer details and predict churn risk
- Save prediction history in SQLite
- Optional PostgreSQL support

## Optional PostgreSQL
Create a database named `customer_churn_db` in pgAdmin, then set this in PowerShell (replace the password):
```powershell
$env:DATABASE_URL="postgresql+psycopg2://postgres:YOUR_PASSWORD@localhost:5432/customer_churn_db"
streamlit run app.py
```
If `DATABASE_URL` is not set, SQLite is used automatically.

**Important:** The included data is synthetic and model results are for learning only. Predictions are estimates, not guarantees.
