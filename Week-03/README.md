# Steel Industry Energy Consumption Prediction

Week 3 Task — **Dimensionality Reduction (PCA) + FastAPI Dashboard**

Predicts steel-industry energy usage (`Usage_kWh`) from 9 process features using a
saved scikit-learn pipeline: **StandardScaler → PCA (9→6 components) → RandomForestRegressor**.

## Project structure

```
Steel_Energy_Project_Final/
├── main.py                     # FastAPI backend (serves dashboard + /predict API)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── week3_energy_model.joblib   # Trained pipeline (scaler + PCA + RandomForest), compressed
├── templates/
│   └── index.html              # Dashboard UI (calls the LOCAL /predict endpoint)
├── static/
│   └── style.css               # Dashboard styles (served at /static/style.css)
└── notebooks/
    └── week3pca.ipynb          # Full notebook, all cells executed with outputs
```

## How to run (Windows / PowerShell)

From inside this folder:

```powershell
# 1. Create a virtual environment (once)
py -m venv venv

# 2. Activate it
.\venv\Scripts\Activate.ps1
#    (If activation is blocked, run once:)
#    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 3. Install dependencies (once)
pip install -r requirements.txt

# 4. Start the server  (WAIT for the "Uvicorn running on ..." line — the model takes a few seconds to load)
uvicorn main:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000** in your browser and click **Predict Now**.

Stop the server with **Ctrl+C** in the terminal.

### API usage (without the dashboard)

```
POST http://127.0.0.1:8000/predict
Content-Type: application/json

{
  "Lagging_Current_Reactive_Power_kVarh": 3.0,
  "Leading_Current_Reactive_Power_kVarh": 0.0,
  "CO2_tCO2": 0.01,
  "Lagging_Current_Power_Factor": 71.0,
  "Leading_Current_Power_Factor": 100.0,
  "NSM": 900,
  "WeekStatus": 0,
  "Day_of_week": 1,
  "Load_Type": 0
}
```
Response: `{"predicted_usage_kWh": 6.529}`

Interactive API docs are available at **http://127.0.0.1:8000/docs**.

## Categorical encodings

The model was trained on label-encoded categoricals (alphabetical order):

| Field       | Values |
|-------------|--------|
| `WeekStatus`  | 0 = Weekday, 1 = Weekend |
| `Day_of_week` | 0 = Friday, 1 = Monday, 2 = Saturday, 3 = Sunday, 4 = Thursday, 5 = Tuesday, 6 = Wednesday |
| `Load_Type`   | 0 = Light_Load, 1 = Maximum_Load, 2 = Medium_Load |

## Troubleshooting

- **`ERR_CONNECTION_REFUSED` in the browser** — the server isn't running/listening.
  Make sure you see `Uvicorn running on http://127.0.0.1:8000` in the terminal *before* opening the page.
- **`[Errno 10048] ... only one usage of each socket address`** — port 8000 is already in
  use by an old run. Find and kill it, or use another port:
  ```powershell
  netstat -ano | findstr ":8000"      # note the PID in the last column
  taskkill /PID <PID> /F
  # or just:  uvicorn main:app --port 8001   (then browse to http://127.0.0.1:8001)
  ```
  Always stop the server with **Ctrl+C** — closing the browser tab does NOT stop it.
- **`InconsistentVersionWarning` about scikit-learn** — harmless. The model was trained on
  scikit-learn 1.6.1; it still predicts correctly on newer versions. To silence it and
  guarantee identical results: `pip install scikit-learn==1.6.1`.

## Notes on `notebooks/week3pca.ipynb`

This is the original Google Colab notebook that trained the model and built the pipeline,
preserved with all cell outputs (PCA variance plot, model comparison, insights). It contains
Colab-specific cells (`google.colab`, proxy/ngrok) that are a record of the Colab workflow and
are **not** needed to run the app locally — the local app is driven entirely by `main.py`.
