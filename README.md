# Customer Segmentation Portal

A small Flask web application that uses a trained **K-Means clustering model** to group mall customers using annual income and spending score, then displays an illustrative marketing idea for the predicted segment.

> **Project scope:** this repository focuses on the Mall Customers dataset and the two features used by the web app. The suggestions are examples for exploration, not validated business recommendations.

## Features

- K-Means clustering with five customer segments.
- `StandardScaler` preprocessing consistent with model training.
- Flask form for entering annual income and spending score.
- Segment descriptions and suggested marketing experiments.
- Exploratory data analysis and clustering experiments in a Jupyter notebook.
- Basic server-side input validation and error logging.
- Automated tests (pytest) run on every push with GitHub Actions.

## Tech stack

- Python
- Flask
- pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn
- Jupyter Notebook
- pytest and GitHub Actions

## Repository structure

```text
customer-segmentation/
├── app.py
├── requirements.txt            # the web app
├── requirements-notebook.txt   # app + notebook libraries
├── requirements-dev.txt        # app + pytest
├── pytest.ini
├── LICENSE
├── .env.example
├── .gitignore
├── Mall_Customers.csv
├── kmeans_model.pkl
├── scaler.pkl
├── .github/
│   └── workflows/
│       └── ci.yml
├── notebooks/
│   └── customer_segmentation_analysis.ipynb
├── static/
│   └── style.css
├── templates/
│   └── index.html
└── tests/
    └── test_app.py
```

## Run locally

### 1. Requirements

Install Python 3.11 or 3.12. Use a virtual environment so dependencies for this project remain separate from other Python projects.

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can run the environment's Python directly with `.\.venv\Scripts\python.exe`.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start the app

From the repository root:

```bash
python app.py
```

Open <http://127.0.0.1:5000> in your browser. Enter annual income in thousands of dollars (`k$`) and a spending score from 1 to 100.

Optional configuration can be set with environment variables. Copy `.env.example` as a reference, but note that this project does not automatically load a `.env` file; set variables in your shell or use a local environment-variable tool. For PowerShell:

```powershell
$env:FLASK_SECRET_KEY = "replace-with-a-long-random-secret"
$env:FLASK_DEBUG = "1"
python app.py
```

Turn debug mode off outside local development. Do not commit real secrets.

## Explore the analysis notebook

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Install the notebook libraries first (`pip install -r requirements-notebook.txt`), then open `notebooks/customer_segmentation_analysis.ipynb` and run the cells in order. The notebook explores the data, scales the selected features, tests clustering approaches, and saves model artifacts. If you retrain the model, make sure the web app's cluster descriptions still match the newly calculated cluster profiles; K-Means cluster IDs can change between model fits.

## Results

- K-Means with **k = 5** (chosen with the elbow method) reaches a silhouette score of **0.555** on the scaled income and spending-score features.
- DBSCAN finds only 2 clusters and flags 8 customers as noise, so K-Means gives the more interpretable segments for this dataset.
- The five K-Means segments are: moderate-income/moderate-spending, high-income/high-spending, lower-income/high-spending, high-income/low-spending and lower-income/low-spending.

## Tests

Install the test dependencies and run pytest from the repository root:

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover the home page, predictions for typical customers, rejection of invalid input (empty values, text, negative income, spending scores outside 1-100, `nan` and `inf`), and a check that each segment name matches its real cluster center. That last test guards against mislabeled segments, because K-Means cluster IDs can change when the model is retrained. The same tests run automatically on every push through GitHub Actions (`.github/workflows/ci.yml`).

## Dataset

`Mall_Customers.csv` contains customer demographic and spending information. The web app uses these columns:

- `Annual Income (k$)` — annual income in thousands of dollars.
- `Spending Score (1-100)` — a score from 1 to 100.

The dataset is included for educational and demonstration purposes. Check the dataset's original source and its license/usage terms before redistributing it in another context.

## Model artifacts and reproducibility

`kmeans_model.pkl` and `scaler.pkl` are serialized scikit-learn objects. The dependency file pins scikit-learn to the version recorded when these artifacts were created to reduce compatibility issues. Pickle/joblib files can execute code when loaded, so only load artifacts you trust. Retraining the model may require updating the artifacts and checking the cluster descriptions in `app.py`.

## Limitations

- K-Means is an exploratory clustering method, not a supervised classifier with ground-truth labels.
- Cluster descriptions are interpretations of group averages and should not be treated as facts about every customer.
- Marketing ideas are illustrative hypotheses; they have not been validated with campaign experiments.
- The model uses only income and spending score, so it omits other factors that may influence customer behavior.
- This is an educational project and is not configured as a production service.

## Possible future improvements

- Add an interactive visualization of cluster profiles and sizes.
- Add a reproducible training script and record the exact dataset source.
- Add accessibility checks and deployment configuration if needed.

## License

The code is released under the MIT License (see `LICENSE`). The license covers the code only. Confirm the dataset's own redistribution terms before reusing `Mall_Customers.csv` elsewhere.
