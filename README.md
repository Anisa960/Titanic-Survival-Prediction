# Titanic Dataset Analysis, Modeling, and Deployment

Contents:
- `Titanic_Assignment.ipynb` — EDA, preprocessing, modeling, model saving.
- `app.py` — Streamlit app that loads `model.pkl` and predicts survival.
- `requirements.txt`

## Dataset
Download Kaggle Titanic `train.csv` and place it next to the notebook or in `./data/`.

## Run
1) `pip install -r requirements.txt`
2) Open notebook and run all cells (saves `model.pkl`).
3) `streamlit run app.py`
