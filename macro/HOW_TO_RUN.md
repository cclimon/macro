# G10 FX Signal Dashboard — How to Run

## Requirements

- Python 3.12
- Bloomberg Terminal open and logged in
- Dependencies installed: `pip install -r requirements.txt`

## Step 1 — Refresh EOD data (needs Bloomberg)

Run from the `macro/` folder:

```
python main.py
```

This fetches live data from Bloomberg, builds all signals, and saves them to
`data/cache/` as parquet files. Takes ~30 seconds.

## Step 2 — Commit the cache to GitHub

Run from the repo root (`GitRepo/`):

```
git add macro/data/cache/
git commit -m "Refresh EOD cache YYYY-MM-DD"
git push
```

## Step 3 — Launch the dashboard locally

Run from the `macro/` folder:

```
streamlit run dashboard/app.py
```

Opens at http://localhost:8501

## Streamlit Cloud (public deployment)

The dashboard is deployed at:
https://share.streamlit.io  ← add your URL here once deployed

To deploy:
1. Go to https://share.streamlit.io and sign in with GitHub
2. Click "New app"
3. Repository: `cclimon/macro`
4. Branch: `main`
5. Main file path: `macro/dashboard/app.py`
6. Click Deploy

No Bloomberg needed on the cloud — it reads from the committed parquet files.
