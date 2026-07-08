from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
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


def plot_quarterly_amounts(tax_data: pd.DataFrame) -> None:
    if tax_data.empty:
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    ax = tax_data.groupby("Quarter")["Amount"].sum().plot(
        kind="bar",
        color="skyblue",
        title="Analysis by Quarter",
    )
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Total Amount")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "forecasting_quarterly_amounts.png", dpi=150)
    plt.close()


def run_forecasting(tax_data: pd.DataFrame) -> RandomForestRegressor:
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

    y_pred = model.predict(x_test)
    accuracy_score = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    predicted_future_impact = model.predict(x_test.iloc[-1:])[0]

    OUTPUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(
        {
            "actual_profit": y_test.reset_index(drop=True),
            "predicted_profit": y_pred,
        }
    ).to_csv(OUTPUT_DIR / "forecasting_predictions.csv", index=False)

    print(f"r2={accuracy_score:.2f}")
    print(f"mae={mae:.2f}")
    print(f"prediction={predicted_future_impact:.2f}")

    return model


if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys

    parser = argparse.ArgumentParser(description="Run the Evolytix financial forecasting model.")
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
        raise SystemExit("Financial forecasting needs at least 5 tax or expense rows.")

    if args.web_json:
        plot_quarterly_amounts(selected_data)
        x = selected_data[["Year", "Month", "Day", "Amount", "Balance"]]
        y = selected_data["Profit"]
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
        )
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy_score = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        predicted_future_impact = model.predict(x_test.iloc[-1:])[0]

        predictions_file = OUTPUT_DIR / "forecasting_predictions.csv"
        pd.DataFrame(
            {
                "actual_profit": y_test.reset_index(drop=True),
                "predicted_profit": y_pred,
            }
        ).to_csv(predictions_file, index=False)

        payload = {
            "summary": {
                "totalRows": int(len(financial_data)),
                "selectedRows": int(len(selected_data)),
                "r2Score": float(accuracy_score),
                "meanAbsoluteError": float(mae),
                "predictedFutureImpact": float(predicted_future_impact),
            },
            "metrics": [
                {"label": "Rows", "value": int(len(financial_data))},
                {"label": "Selected", "value": int(len(selected_data))},
                {"label": "R2 score", "value": float(accuracy_score), "format": "decimal"},
                {"label": "MAE", "value": float(mae), "format": "currency"},
                {"label": "Prediction", "value": float(predicted_future_impact), "format": "currency"},
            ],
            "tableTitle": "Prediction sample",
            "flaggedRows": pd.DataFrame(
                {
                    "actual_profit": y_test.reset_index(drop=True),
                    "predicted_profit": y_pred,
                }
            ).head(10).to_dict(orient="records"),
            "plotFile": "forecasting_quarterly_amounts.png",
            "notes": [
                "Forecasting uses tax rows when available; otherwise it uses expense rows.",
                "The model predicts Profit from date features, Amount, and Balance.",
            ],
        }
        print("WEB_JSON:" + json.dumps(payload, default=str))
        sys.exit(0)

    plot_quarterly_amounts(selected_data)
    run_forecasting(selected_data)
