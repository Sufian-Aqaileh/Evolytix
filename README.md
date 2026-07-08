# Evolytix Static Site

Evolytix is a static web prototype with a small Python API for running the auditing anomaly model from the Analysis page.

## Project Structure

```text
backend/
  test_adaptive_auditing.py  # auditing worker used by the web server
  02_financial_forecasting.py
  03_financial_optimization_advisory.py
public/
  assets/
    script.js
    styles.css
  *.html                    # static pages served by server.py
server.py                   # local web server and audit API
requirements.txt
```

## Run Locally

Install dependencies if needed:

```bash
python -m pip install -r requirements.txt
```

Start the site and API:

```bash
python server.py
```

Open:

```text
http://127.0.0.1:8000
```

The Analysis page accepts CSV files with these columns:

```text
Debit, Credit, Amount, Category, Transaction_Type, Payment_Method, Balance, Previous_Balance
```

## Notes

The web server calls each model script with `--web-json` as a worker process. The worker writes any matplotlib output to `public/generated/...` and returns JSON for the Analysis page.

Available models:

- `auditing`: runs `backend/test_adaptive_auditing.py`
- `forecasting`: runs `backend/02_financial_forecasting.py`
- `advisory`: runs `backend/03_financial_optimization_advisory.py`
