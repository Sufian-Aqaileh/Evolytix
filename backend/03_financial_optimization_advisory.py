from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import linprog
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


DATA_PATH = "financial_accounting_cleaned.csv"
OUTPUT_DIR = Path("outputs")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if str(path).lower().endswith(".parquet"):
        data = pd.read_parquet(path)
    else:
        data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    return data


def prepare_tax_or_expense_data(data: pd.DataFrame) -> pd.DataFrame:
    tax_data = data[data["Category"].str.contains("Tax", case=False, na=False)].copy()
    if tax_data.empty:
        tax_data = data[data["Category"] == "Expense"].copy()

    tax_data["Date"] = pd.to_datetime(tax_data["Date"])
    tax_data["Year"] = tax_data["Date"].dt.year
    tax_data["Month"] = tax_data["Date"].dt.month
    tax_data["Day"] = tax_data["Date"].dt.day
    tax_data["Quarter"] = tax_data["Date"].dt.quarter
    tax_data["Profit"] = tax_data["Credit"] - tax_data["Debit"]
    return tax_data


def forecast_financial_impact(tax_data: pd.DataFrame) -> float:
    x = tax_data[["Year", "Month", "Day", "Amount", "Balance"]]
    y = tax_data["Profit"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(x_train, y_train)

    predicted_future_impact = model.predict(x_test.iloc[-1:])[0]

    return predicted_future_impact


def optimize_financial_target(predicted_future_impact: float):
    c = [predicted_future_impact]
    a_ub = [[1]]
    b_ub = [0]
    result = linprog(c, A_ub=a_ub, b_ub=b_ub, method="highs")

    print(f"success={result.success}")
    print(f"target={result.x[0]:.2f}")
    print(f"forecast={predicted_future_impact:.2f}")
    return result


if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys

    parser = argparse.ArgumentParser(description="Run the Evolytix financial optimization advisory model.")
    parser.add_argument("--web-json", action="store_true", help="Emit a final JSON payload for the web server.")
    parser.add_argument("--csv", default=DATA_PATH, help="CSV file to analyze.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory where outputs should be written.")
    args = parser.parse_args()

    if args.web_json:
        OUTPUT_DIR = Path(args.output_dir).resolve()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        os.chdir(OUTPUT_DIR)
        plt.show = lambda *show_args, **show_kwargs: None

    financial_data = load_data(args.csv)
    selected_data = prepare_tax_or_expense_data(financial_data)
    if len(selected_data.index) < 5:
        raise SystemExit("Financial optimization advisory needs at least 5 tax or expense rows.")

    future_impact = forecast_financial_impact(selected_data)
    result = optimize_financial_target(future_impact)

    if args.web_json:
        chart_path = OUTPUT_DIR / "optimization_advisory.png"
        plt.figure(figsize=(8, 4.5))
        plt.bar(
            ["Forecast impact", "Optimized target"],
            [float(future_impact), float(result.x[0])],
            color=["#4f8cff", "#68d391"],
        )
        plt.title("Financial Optimization Advisory")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150)
        plt.close()

        payload = {
            "summary": {
                "totalRows": int(len(financial_data)),
                "selectedRows": int(len(selected_data)),
                "forecastImpact": float(future_impact),
                "optimizedTarget": float(result.x[0]),
                "optimizationSuccess": bool(result.success),
            },
            "metrics": [
                {"label": "Rows", "value": int(len(financial_data))},
                {"label": "Selected", "value": int(len(selected_data))},
                {"label": "Forecast", "value": float(future_impact), "format": "currency"},
                {"label": "Target", "value": float(result.x[0]), "format": "currency"},
                {"label": "Success", "value": "Yes" if result.success else "No"},
            ],
            "tableTitle": "Advisory summary",
            "flaggedRows": [
                {
                    "forecast_impact": float(future_impact),
                    "optimized_target": float(result.x[0]),
                    "success": bool(result.success),
                    "status": int(result.status),
                    "message": result.message,
                }
            ],
            "plotFile": "optimization_advisory.png",
            "notes": [
                "The advisory model forecasts financial impact from tax or expense rows.",
                "Linear programming is used to calculate the optimization target.",
            ],
        }
        print("WEB_JSON:" + json.dumps(payload, default=str))
        sys.exit(0)
