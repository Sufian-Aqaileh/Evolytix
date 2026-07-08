import os
import re
import sys

import numpy as np
import pandas as pd


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_FILE_NAME = "financial_accounting.csv"
DEFAULT_CLEANED_OUTPUT_FILE_NAME = "financial_accounting_cleaned.parquet"
DEFAULT_REPORT_FILE_NAME = "data_quality_report.csv"
PARQUET_ENGINE = "pyarrow"


def configure_standard_streams():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")
            except (AttributeError, OSError, ValueError):
                pass


def safe_print(message):
    try:
        print(message)
    except UnicodeEncodeError:
        stream = sys.stdout
        encoding = getattr(stream, "encoding", None) or "utf-8"
        encoded_message = f"{message}\n".encode(encoding, errors="backslashreplace")
        buffer = getattr(stream, "buffer", None)
        if buffer is not None:
            buffer.write(encoded_message)
            buffer.flush()
            return
        stream.write(encoded_message.decode(encoding))
        stream.flush()


configure_standard_streams()

REQUIRED_COLUMNS = [
    "Date",
    "Account",
    "Description",
    "Debit",
    "Credit",
    "Category",
    "Transaction_Type",
]

OPTIONAL_COLUMNS = ["Customer_Vendor", "Payment_Method", "Reference"]

REDUNDANT_COLUMNS = [
    "Counterparty_Name",
    "Payment_Channel",
    "Document_Reference",
    "Original_Row_Order",
]

TEXT_COLUMNS = [
    "Account",
    "Description",
    "Transaction_Type",
    "Customer_Vendor",
    "Payment_Method",
    "Reference",
]

COLUMN_ALIASES = {
    # Date
    "date": "Date",
    "transaction_date": "Date",
    "posting_date": "Date",
    "entry_date": "Date",
    "journal_date": "Date",
    "document_date": "Date",
    "invoice_date": "Date",
    "voucher_date": "Date",
    "value_date": "Date",

    # Account
    "account": "Account",
    "account_name": "Account",
    "account_title": "Account",
    "ledger_account": "Account",
    "gl_account": "Account",
    "gl_account_name": "Account",
    "general_ledger_account": "Account",

    # Description
    "description": "Description",
    "transaction_description": "Description",
    "entry_description": "Description",
    "journal_description": "Description",
    "line_description": "Description",

    # Debit
    "debit": "Debit",
    "debit_amount": "Debit",
    "amount_debit": "Debit",
    "debit_value": "Debit",
    "dr_amount": "Debit",

    # Credit
    "credit": "Credit",
    "credit_amount": "Credit",
    "amount_credit": "Credit",
    "credit_value": "Credit",
    "cr_amount": "Credit",

    # Category
    "category": "Category",
    "account_category": "Category",
    "financial_category": "Category",

    # Transaction_Type
    "transaction_type": "Transaction_Type",
    "transactiontype": "Transaction_Type",
    "txn_type": "Transaction_Type",
    "trx_type": "Transaction_Type",
    "transaction_kind": "Transaction_Type",

    # Customer_Vendor
    "customer_vendor": "Customer_Vendor",
    "customervendor": "Customer_Vendor",
    "customer_vendor_name": "Customer_Vendor",
    "customer_or_vendor": "Customer_Vendor",
    "customer_name": "Customer_Vendor",
    "vendor_name": "Customer_Vendor",
    "supplier_name": "Customer_Vendor",
    "client_name": "Customer_Vendor",

    # Payment_Method
    "payment_method": "Payment_Method",
    "paymentmethod": "Payment_Method",
    "method_of_payment": "Payment_Method",
    "pay_method": "Payment_Method",

    # Reference
    "reference": "Reference",
    "reference_no": "Reference",
    "reference_number": "Reference",
    "ref_no": "Reference",
    "ref_number": "Reference",
    "document_reference": "Reference",
    "voucher_no": "Reference",
    "voucher_number": "Reference",
    "invoice_no": "Reference",
    "invoice_number": "Reference",
    "receipt_no": "Reference",
    "receipt_number": "Reference",
    "journal_no": "Reference",
    "journal_number": "Reference",
    "source_reference": "Reference",

    # Redundant columns still recognized for safe dropping if present
    "payment_channel": "Payment_Channel",
    "original_row_order": "Original_Row_Order",
}

CATEGORY_MAP = {
    "asset": "Assets",
    "assets": "Assets",
    "current asset": "Assets",
    "current assets": "Assets",
    "fixed asset": "Assets",
    "fixed assets": "Assets",
    "non current asset": "Assets",
    "non current assets": "Assets",
    "receivable": "Assets",
    "receivables": "Assets",
    "accounts receivable": "Assets",
    "inventory": "Assets",
    "prepaid": "Assets",
    "prepayments": "Assets",
    "revenue": "Revenue",
    "revenues": "Revenue",
    "sales revenue": "Revenue",
    "service revenue": "Revenue",
    "operating revenue": "Revenue",
    "expense": "Expense",
    "expenses": "Expense",
    "operating expense": "Expense",
    "operating expenses": "Expense",
    "cogs": "Expense",
    "cost of goods sold": "Expense",
    "cost of sales": "Expense",
    "liability": "Liability",
    "liabilities": "Liability",
    "current liability": "Liability",
    "current liabilities": "Liability",
    "non current liability": "Liability",
    "non current liabilities": "Liability",
    "payable": "Liability",
    "payables": "Liability",
    "accounts payable": "Liability",
    "accrued liability": "Liability",
    "accrued liabilities": "Liability",
    "tax payable": "Liability",
    "deferred revenue": "Liability",
    "unearned revenue": "Liability",
    "equity": "Equity",
    "equities": "Equity",
    "owner equity": "Equity",
    "owners equity": "Equity",
    "shareholder equity": "Equity",
    "retained earnings": "Equity",
}

EXPECTED_SIDE_MAP = {
    "Assets": "Debit",
    "Expense": "Debit",
    "Revenue": "Credit",
    "Liability": "Credit",
    "Equity": "Credit",
}

SIGNED_AMOUNT_MAP = {
    "Assets": 1,
    "Revenue": 1,
    "Equity": 1,
    "Expense": -1,
    "Liability": -1,
}

MISSING_TOKENS = {"", "nan", "none", "null", "n/a", "na"}

FINAL_COLUMN_ORDER = [
    "Date",
    "Account",
    "Description",
    "Debit",
    "Credit",
    "Category",
    "Transaction_Type",
    "Customer_Vendor",
    "Payment_Method",
    "Reference",
    "Validation_Flag",
    "Amount",
    "Balance",
    "Previous_Balance",
    "Year",
    "Month",
    "Quarter",
]

REPORT_COLUMNS = [
    "Source_Row",
    "Issue_Code",
    "Severity",
    "Column_Name",
    "Current_Value",
    "Account",
    "Description",
    "Date",
    "Details",
    "Suggested_Action",
]

SEVERITY_RANK = {
    "low": 1,
    "medium": 2,
    "high": 3,
}

SEVERITY_BY_RANK = {
    1: "low",
    2: "medium",
    3: "high",
}


def join_unique_values(values):
    unique_values = []
    seen_values = set()

    for value in values:
        if pd.isna(value):
            continue

        if isinstance(value, pd.Timestamp):
            text_value = value.strftime("%Y-%m-%d")
        else:
            text_value = str(value).strip()

        if not text_value or text_value in seen_values:
            continue

        unique_values.append(text_value)
        seen_values.add(text_value)

    return "; ".join(unique_values)


def summarize_current_values(values):
    return join_unique_values(values)


def summarize_quality_report(detailed_report_df):
    if detailed_report_df.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    report_df = detailed_report_df.copy()
    for column in REPORT_COLUMNS:
        if column not in report_df.columns:
            report_df[column] = ""

    report_df = report_df[REPORT_COLUMNS].fillna("")

    def summarize_severity(values):
        severity_rank = 0
        for value in values:
            severity_rank = max(severity_rank, SEVERITY_RANK.get(str(value).lower(), 0))
        return SEVERITY_BY_RANK.get(severity_rank, join_unique_values(values))

    summarized_report_df = (
        report_df.groupby("Source_Row", sort=False, dropna=False)
        .agg(
            {
                "Issue_Code": join_unique_values,
                "Severity": summarize_severity,
                "Column_Name": join_unique_values,
                "Current_Value": summarize_current_values,
                "Account": join_unique_values,
                "Description": join_unique_values,
                "Date": join_unique_values,
                "Details": join_unique_values,
                "Suggested_Action": join_unique_values,
            }
        )
        .reset_index()
    )

    return summarized_report_df[REPORT_COLUMNS]


def resolve_path(file_name):
    if os.path.isabs(file_name):
        return file_name
    return os.path.join(PROJECT_DIR, file_name)


def build_default_output_names(input_file):
    input_name = os.path.basename(input_file)
    input_stem, _ = os.path.splitext(input_name)

    if input_name.lower() == DEFAULT_INPUT_FILE_NAME.lower():
        return DEFAULT_CLEANED_OUTPUT_FILE_NAME, DEFAULT_REPORT_FILE_NAME

    return f"{input_stem}_cleaned.parquet", f"{input_stem}_data_quality_report.csv"


def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if file_path.lower().endswith(".csv"):
        return pd.read_csv(file_path, dtype=str, keep_default_na=False)

    if file_path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path, dtype=str)

    if file_path.lower().endswith(".parquet"):
        return pd.read_parquet(file_path, engine=PARQUET_ENGINE)

    raise ValueError("Unsupported file format. Use CSV, Excel, or Parquet.")


def normalize_column_name(column_name):
    cleaned_name = re.sub(r"[^a-z0-9]+", "_", str(column_name).strip().lower())
    return cleaned_name.strip("_")


def standardize_column_names(df):
    df = df.copy()
    rename_map = {}
    used_names = set()

    for column in df.columns:
        normalized_name = normalize_column_name(column)
        canonical_name = COLUMN_ALIASES.get(normalized_name, str(column).strip())

        if canonical_name in used_names:
            suffix = 2
            updated_name = f"{canonical_name}_{suffix}"
            while updated_name in used_names:
                suffix += 1
                updated_name = f"{canonical_name}_{suffix}"
            canonical_name = updated_name

        rename_map[column] = canonical_name
        used_names.add(canonical_name)

    return df.rename(columns=rename_map)


def validate_required_columns(df):
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def ensure_optional_columns(df):
    df = df.copy()
    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    return df


def add_source_row(df):
    df = df.copy()
    df["_Source_Row"] = np.arange(1, len(df) + 1)
    return df


def drop_redundant_columns(df):
    df = df.copy()
    columns_to_drop = [column for column in REDUNDANT_COLUMNS if column in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    return df


def clean_text_value(value):
    if pd.isna(value):
        return np.nan

    text = str(value).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if text.lower() in MISSING_TOKENS:
        return np.nan

    return text


def normalize_category_key(value):
    cleaned_value = clean_text_value(value)
    if pd.isna(cleaned_value):
        return np.nan

    normalized_value = re.sub(r"[^a-z0-9]+", " ", str(cleaned_value).lower()).strip()
    normalized_value = re.sub(r"\s+", " ", normalized_value)
    return normalized_value or np.nan


def clean_text_columns(df):
    df = df.copy()
    for column in TEXT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].apply(clean_text_value)
    return df


def has_time_component(date_text):
    if pd.isna(date_text):
        return False

    text = str(date_text).strip()
    if text.lower() in MISSING_TOKENS:
        return False

    time_pattern = (
        r"(?:[ T])\d{1,2}:\d{2}"
        r"(?::\d{2}(?:\.\d+)?)?"
        r"(?:\s?(?:AM|PM|am|pm))?"
        r"(?:Z|[+-]\d{2}:?\d{2})?$"
    )
    return bool(re.search(time_pattern, text))


def parse_numeric_series(series):
    cleaned_text = series.apply(clean_text_value)

    normalized_text = cleaned_text.copy()
    normalized_text = normalized_text.astype("object")

    present_mask = normalized_text.notna()
    normalized_text.loc[present_mask] = (
        normalized_text.loc[present_mask]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
    )

    parentheses_mask = normalized_text.astype(str).str.match(r"^\(.*\)$", na=False)
    normalized_text.loc[parentheses_mask] = (
        "-" + normalized_text.loc[parentheses_mask].astype(str).str[1:-1]
    )

    numeric_values = pd.to_numeric(normalized_text, errors="coerce")

    missing_mask = cleaned_text.isna()
    invalid_mask = (~missing_mask) & numeric_values.isna()
    negative_mask = numeric_values < 0
    zero_mask = numeric_values.eq(0)
    positive_mask = numeric_values > 0

    return numeric_values, {
        "missing": missing_mask,
        "invalid": invalid_mask,
        "negative": negative_mask.fillna(False),
        "zero": zero_mask.fillna(False),
        "positive": positive_mask.fillna(False),
    }


def convert_date_column(df):
    df = df.copy()

    raw_date = df["Date"].apply(clean_text_value)
    raw_time_present = raw_date.apply(has_time_component)
    converted_date = raw_date.apply(
        lambda value: pd.to_datetime(value, errors="coerce") if pd.notna(value) else pd.NaT
    )
    converted_date = pd.to_datetime(converted_date, errors="coerce")
    missing_mask = raw_date.isna()
    invalid_mask = (~missing_mask) & converted_date.isna()
    future_mask = converted_date > pd.Timestamp.today().normalize()
    time_present_mask = (~missing_mask) & raw_time_present

    converted_date = converted_date.dt.normalize()

    df["_Raw_Date_Text"] = raw_date
    df["Date"] = converted_date

    return df, {
        "missing": missing_mask,
        "invalid": invalid_mask,
        "future": future_mask.fillna(False),
        "time_present": time_present_mask.fillna(False),
    }


def normalize_category(df):
    df = df.copy()

    raw_category = df["Category"].apply(clean_text_value)
    normalized_category = raw_category.apply(normalize_category_key).map(CATEGORY_MAP)

    missing_mask = raw_category.isna()
    unknown_mask = (~missing_mask) & normalized_category.isna()

    df["Category"] = normalized_category.where(normalized_category.notna(), raw_category)
    df["_Expected_Side"] = df["Category"].map(EXPECTED_SIDE_MAP)

    return df, {
        "missing": missing_mask,
        "unknown": unknown_mask,
    }


def add_issue_records(
    report_rows,
    df,
    mask,
    issue_code,
    severity,
    column_name,
    details,
    suggested_action,
    value_column=None,
):
    issue_mask = mask.fillna(False) if isinstance(mask, pd.Series) else mask
    if not issue_mask.any():
        return

    for _, row in df.loc[issue_mask].iterrows():
        current_value = ""
        if value_column is not None:
            value = row[value_column]
            current_value = "" if pd.isna(value) else str(value)
            if value_column == "Reference" and current_value:
                current_value = f"Reference={current_value}"

        report_rows.append(
            {
                "Source_Row": int(row["_Source_Row"]),
                "Issue_Code": issue_code,
                "Severity": severity,
                "Column_Name": column_name,
                "Current_Value": current_value,
                "Account": "" if pd.isna(row.get("Account")) else str(row.get("Account")),
                "Description": "" if pd.isna(row.get("Description")) else str(row.get("Description")),
                "Date": (
                    ""
                    if pd.isna(row.get("Date"))
                    else (
                        row.get("Date").strftime("%Y-%m-%d")
                        if isinstance(row.get("Date"), pd.Timestamp)
                        else str(row.get("Date"))
                    )
                ),
                "Details": details,
                "Suggested_Action": suggested_action,
            }
        )


def remove_exact_duplicates(df, report_rows):
    df = df.copy()
    compare_columns = [column for column in df.columns if column != "_Source_Row"]
    duplicate_mask = df.duplicated(subset=compare_columns, keep="first")

    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=duplicate_mask,
        issue_code="exact_duplicate_removed",
        severity="medium",
        column_name="row",
        details="This row matched an earlier row exactly after safe cleaning.",
        suggested_action="Review the removed duplicate against the original source document.",
    )

    return df.loc[~duplicate_mask].reset_index(drop=True)


def normalize_mirrored_amount_rows(df, report_rows):
    df = df.copy()

    same_positive_amount = (
        df["Debit"].notna()
        & df["Credit"].notna()
        & (df["Debit"] > 0)
        & (df["Credit"] > 0)
        & np.isclose(df["Debit"], df["Credit"])
    )
    known_expected_side = df["_Expected_Side"].notna()
    normalize_mask = same_positive_amount & known_expected_side

    debit_side_mask = normalize_mask & df["_Expected_Side"].eq("Debit")
    credit_side_mask = normalize_mask & df["_Expected_Side"].eq("Credit")

    df.loc[debit_side_mask, "Credit"] = 0.0
    df.loc[credit_side_mask, "Debit"] = 0.0

    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=normalize_mask,
        issue_code="mirrored_debit_credit_normalized",
        severity="low",
        column_name="Debit/Credit",
        details="Debit and Credit had the same positive amount, so the value was kept only on the expected side based on Category.",
        suggested_action="Review if this dataset intentionally mirrors both columns. If yes, this normalization is expected.",
    )

    return df


def create_amount_column(df):
    df = df.copy()

    amount = pd.Series(np.nan, index=df.index, dtype="float64")

    debit_positive = df["Debit"] > 0
    credit_positive = df["Credit"] > 0
    both_zero = df["Debit"].eq(0) & df["Credit"].eq(0)

    amount.loc[debit_positive] = df.loc[debit_positive, "Debit"]
    amount.loc[~debit_positive & credit_positive] = df.loc[~debit_positive & credit_positive, "Credit"]
    amount.loc[amount.isna() & both_zero] = 0.0

    df["Amount"] = amount
    return df


def create_signed_amount_column(df):
    df = df.copy()
    sign_values = df["Category"].map(SIGNED_AMOUNT_MAP)
    df["Signed_Amount"] = df["Amount"] * sign_values
    return df


def merge_signed_amount_into_amount(df):
    df = df.copy()

    if "Signed_Amount" in df.columns:
        df["Amount"] = df["Signed_Amount"]
        df = df.drop(columns=["Signed_Amount"])

    return df


def sort_accounting_data(df):
    df = df.copy()
    df = df.sort_values(
        by=["Date", "Account", "_Source_Row"],
        ascending=[True, True, True],
        kind="mergesort",
        na_position="last",
    )
    return df.reset_index(drop=True)


def calculate_balance_columns(df):
    df = df.copy()

    previous_balances = pd.Series(np.nan, index=df.index, dtype="float64")
    balances = pd.Series(np.nan, index=df.index, dtype="float64")

    for _, group in df.groupby("Account", dropna=False, sort=False):
        running_balance = 0.0

        for row_index, signed_amount in group["Signed_Amount"].items():
            previous_balances.loc[row_index] = running_balance

            if pd.notna(signed_amount):
                running_balance += signed_amount
                balances.loc[row_index] = running_balance
            else:
                balances.loc[row_index] = np.nan

    df["Previous_Balance"] = previous_balances
    df["Balance"] = balances

    return df


def create_time_columns(df):
    df = df.copy()
    df["Year"] = df["Date"].dt.year.astype("Int64")
    df["Month"] = df["Date"].dt.to_period("M").astype(str).replace("NaT", "")
    df["Quarter"] = df["Date"].dt.to_period("Q").astype(str).replace("NaT", "")
    return df


def add_amount_and_reference_checks(df, report_rows):
    debit_missing = df["Debit"].isna()
    credit_missing = df["Credit"].isna()
    both_missing = debit_missing & credit_missing
    both_zero = df["Debit"].eq(0) & df["Credit"].eq(0)
    both_positive = (df["Debit"] > 0) & (df["Credit"] > 0)
    both_positive_different = both_positive & ~np.isclose(df["Debit"], df["Credit"])

    side_conflict_mask = (
        ((df["Debit"] > 0) & df["Credit"].fillna(0).eq(0) & df["_Expected_Side"].eq("Credit"))
        | ((df["Credit"] > 0) & df["Debit"].fillna(0).eq(0) & df["_Expected_Side"].eq("Debit"))
    )

    missing_reference_mask = df["Reference"].isna()
    duplicate_reference_mask = (
        df["Reference"].notna()
        & df.duplicated(subset=["Account", "Reference", "Date"], keep=False)
    )

    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=both_missing,
        issue_code="both_amount_fields_missing",
        severity="high",
        column_name="Debit/Credit",
        details="Both Debit and Credit are missing after safe numeric parsing.",
        suggested_action="Check the source entry and fill the correct side before financial use.",
    )
    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=both_zero,
        issue_code="both_amount_fields_zero",
        severity="medium",
        column_name="Debit/Credit",
        details="Both Debit and Credit are true zero values.",
        suggested_action="Confirm whether this row should remain as a zero-value accounting line.",
    )
    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=both_positive_different,
        issue_code="both_amount_fields_filled_different_values",
        severity="high",
        column_name="Debit/Credit",
        details="Both Debit and Credit contain different positive values, so no automatic correction was applied.",
        suggested_action="Review the row manually against the original accounting entry.",
    )
    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=side_conflict_mask,
        issue_code="amount_side_conflicts_with_category",
        severity="medium",
        column_name="Debit/Credit",
        details="The populated amount side does not match the expected natural side of the Category.",
        suggested_action="Review whether the Category or the Debit/Credit side is incorrect.",
    )
    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=missing_reference_mask,
        issue_code="missing_reference",
        severity="low",
        column_name="Reference",
        details="Reference is missing.",
        suggested_action="Add the source document or journal reference if available.",
        value_column="Reference",
    )
    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=duplicate_reference_mask,
        issue_code="duplicate_reference_same_account_date",
        severity="medium",
        column_name="Reference",
        details="The same Reference appears more than once for the same Account and Date.",
        suggested_action="Check whether the repeated reference is legitimate or a possible duplicate posting.",
        value_column="Reference",
    )


def add_required_field_checks(df, report_rows, date_status, category_status, debit_status, credit_status):
    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=df["Account"].isna(),
        issue_code="missing_account",
        severity="high",
        column_name="Account",
        details="Account is missing.",
        suggested_action="Provide the accounting account before using this row.",
        value_column="Account",
    )
    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=df["Description"].isna(),
        issue_code="missing_description",
        severity="medium",
        column_name="Description",
        details="Description is missing.",
        suggested_action="Add a meaningful transaction description for audit traceability.",
        value_column="Description",
    )
    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=df["Transaction_Type"].isna(),
        issue_code="missing_transaction_type",
        severity="medium",
        column_name="Transaction_Type",
        details="Transaction_Type is missing.",
        suggested_action="Fill the transaction type if it is known.",
        value_column="Transaction_Type",
    )
    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=date_status["missing"],
        issue_code="missing_date",
        severity="high",
        column_name="Date",
        details="Date is missing.",
        suggested_action="Add the transaction date before using this row.",
        value_column="_Raw_Date_Text",
    )
    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=date_status["invalid"],
        issue_code="invalid_date",
        severity="high",
        column_name="Date",
        details="Date could not be converted using pandas with errors='coerce'.",
        suggested_action="Correct the source date value and reload the data.",
        value_column="_Raw_Date_Text",
    )
    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=date_status["future"],
        issue_code="future_date",
        severity="medium",
        column_name="Date",
        details="Date is in the future.",
        suggested_action="Confirm whether the posting date is correct.",
        value_column="_Raw_Date_Text",
    )
    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=date_status["time_present"],
        issue_code="date_contains_time_component",
        severity="low",
        column_name="Date",
        details="The source Date included a time component. The cleaned output keeps only the date part in YYYY-MM-DD format.",
        suggested_action="Review whether the source system should store date-only values or whether the removed time should be preserved elsewhere.",
        value_column="_Raw_Date_Text",
    )
    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=category_status["missing"],
        issue_code="missing_category",
        severity="high",
        column_name="Category",
        details="Category is missing.",
        suggested_action="Provide a valid accounting category.",
        value_column="Category",
    )
    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=category_status["unknown"],
        issue_code="unknown_category",
        severity="high",
        column_name="Category",
        details="Category is not one of the supported accounting categories: Assets, Revenue, Expense, Liability, Equity.",
        suggested_action="Map the Category to a supported accounting category before analysis.",
        value_column="Category",
    )
    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=debit_status["invalid"],
        issue_code="invalid_debit",
        severity="high",
        column_name="Debit",
        details="Debit could not be converted to a numeric value.",
        suggested_action="Fix the raw Debit value rather than replacing it with zero.",
        value_column="Debit",
    )
    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=credit_status["invalid"],
        issue_code="invalid_credit",
        severity="high",
        column_name="Credit",
        details="Credit could not be converted to a numeric value.",
        suggested_action="Fix the raw Credit value rather than replacing it with zero.",
        value_column="Credit",
    )
    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=debit_status["negative"],
        issue_code="negative_debit",
        severity="high",
        column_name="Debit",
        details="Debit is negative.",
        suggested_action="Review whether the sign or the side of the entry is wrong.",
        value_column="Debit",
    )
    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=credit_status["negative"],
        issue_code="negative_credit",
        severity="high",
        column_name="Credit",
        details="Credit is negative.",
        suggested_action="Review whether the sign or the side of the entry is wrong.",
        value_column="Credit",
    )


def add_balance_and_outlier_checks(df, report_rows):
    signed_amount_missing_mask = df["Signed_Amount"].isna() & df["Amount"].notna()
    balance_missing_mask = df["Balance"].isna() & df["Signed_Amount"].isna()
    outlier_mask = pd.Series(False, index=df.index)

    for _, group in df.groupby("Account", dropna=False):
        valid_amounts = group["Amount"].dropna()
        if len(valid_amounts) < 10:
            continue

        q1 = valid_amounts.quantile(0.25)
        q3 = valid_amounts.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower_bound = q1 - (3 * iqr)
        upper_bound = q3 + (3 * iqr)

        group_outliers = group["Amount"].notna() & (
            (group["Amount"] < lower_bound) | (group["Amount"] > upper_bound)
        )
        outlier_mask.loc[group.index] = group_outliers

    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=signed_amount_missing_mask,
        issue_code="signed_amount_not_computed",
        severity="high",
        column_name="Signed_Amount",
        details="Signed_Amount could not be computed because Amount or Category is not reliable.",
        suggested_action="Fix the underlying amount or category issue before using balances.",
        value_column="Signed_Amount",
    )
    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=balance_missing_mask,
        issue_code="balance_not_computed_for_row",
        severity="high",
        column_name="Balance",
        details="Balance was not computed for this row because Signed_Amount is missing.",
        suggested_action="Correct the row and recalculate balances.",
        value_column="Balance",
    )
    add_issue_records(
        report_rows=report_rows,
        df=df,
        mask=outlier_mask,
        issue_code="possible_amount_outlier",
        severity="low",
        column_name="Amount",
        details="Amount is unusually high or low compared with other rows in the same Account.",
        suggested_action="Review the source document to confirm the amount is valid.",
        value_column="Amount",
    )


def add_column_summary_issues(df, report_rows):
    review_columns = [
        "Date",
        "Account",
        "Category",
        "Debit",
        "Credit",
        "Reference",
        "Customer_Vendor",
        "Payment_Method",
    ]

    for column in review_columns:
        if column not in df.columns:
            continue

        missing_rate = df[column].isna().mean()
        if missing_rate >= 0.10:
            report_rows.append(
                {
                    "Source_Row": "",
                    "Issue_Code": "column_missing_rate_high",
                    "Severity": "low",
                    "Column_Name": column,
                    "Current_Value": f"{missing_rate:.2%}",
                    "Account": "",
                    "Description": "",
                    "Date": "",
                    "Details": f"{column} has a high missing rate in the cleaned dataset.",
                    "Suggested_Action": "Review whether the source system is systematically leaving this field blank.",
                }
            )


def build_validation_flag(df, report_df):
    if report_df.empty:
        return pd.Series("OK", index=df.index)

    row_level_report = report_df[report_df["Source_Row"].astype(str).str.strip().ne("")]
    grouped_flags = (
        row_level_report.groupby("Source_Row")["Issue_Code"]
        .apply(lambda values: "; ".join(dict.fromkeys(values)))
        .to_dict()
    )

    return df["_Source_Row"].map(grouped_flags).fillna("OK")


def format_output_columns(df):
    df = df.copy()

    valid_dates = df["Date"].notna()
    date_as_text = pd.Series("", index=df.index, dtype="object")
    date_as_text.loc[valid_dates] = df.loc[valid_dates, "Date"].dt.strftime("%Y-%m-%d")
    df["Date"] = date_as_text

    df["Year"] = df["Year"].astype(str).replace("<NA>", "")

    for column in ["Account", "Description", "Category", "Transaction_Type", "Customer_Vendor", "Payment_Method", "Reference"]:
        if column in df.columns:
            df[column] = df[column].fillna("")

    return df


def order_final_columns(df):
    ordered_columns = [column for column in FINAL_COLUMN_ORDER if column in df.columns]
    remaining_columns = [
        column
        for column in df.columns
        if column not in ordered_columns and not column.startswith("_")
    ]
    return df[ordered_columns + remaining_columns]


def prepare_accounting_data(df):
    report_rows = []

    df = standardize_column_names(df)
    validate_required_columns(df)
    df = ensure_optional_columns(df)
    df = add_source_row(df)
    df = drop_redundant_columns(df)
    df = clean_text_columns(df)
    df, date_status = convert_date_column(df)
    df["Debit"], debit_status = parse_numeric_series(df["Debit"])
    df["Credit"], credit_status = parse_numeric_series(df["Credit"])
    df, category_status = normalize_category(df)

    add_required_field_checks(df, report_rows, date_status, category_status, debit_status, credit_status)

    df = remove_exact_duplicates(df, report_rows)
    df = normalize_mirrored_amount_rows(df, report_rows)
    df = create_amount_column(df)
    df = create_signed_amount_column(df)
    df = sort_accounting_data(df)
    df = calculate_balance_columns(df)
    df = create_time_columns(df)

    add_amount_and_reference_checks(df, report_rows)
    add_balance_and_outlier_checks(df, report_rows)

    detailed_report_df = pd.DataFrame(report_rows, columns=REPORT_COLUMNS)
    summarized_report_df = summarize_quality_report(detailed_report_df)

    df["Validation_Flag"] = build_validation_flag(df, detailed_report_df)
    df = merge_signed_amount_into_amount(df)
    df = format_output_columns(df)
    df = order_final_columns(df)

    df = df.drop(columns=[column for column in df.columns if column.startswith("_")], errors="ignore")

    return df, summarized_report_df


def build_csv_output_path(output_file_name):
    output_path = resolve_path(output_file_name)
    output_root, _ = os.path.splitext(output_path)
    return f"{output_root}.csv"


def save_dataframe(df, output_file_name):
    output_path = resolve_path(output_file_name)
    if output_path.lower().endswith(".parquet"):
        df.to_parquet(output_path, index=False, engine=PARQUET_ENGINE)
    else:
        df.to_csv(output_path, index=False)
    return output_path


def save_csv_copy(df, output_file_name):
    csv_output_path = build_csv_output_path(output_file_name)
    df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    return csv_output_path


def save_cleaned_data(df, output_file_name):
    return save_dataframe(df, output_file_name)


def save_quality_report(report_df, output_file_name):
    return save_dataframe(report_df, output_file_name)


def main(input_file=DEFAULT_INPUT_FILE_NAME, cleaned_output_file=None, report_output_file=None):
    input_path = resolve_path(input_file)
    default_cleaned_name, default_report_name = build_default_output_names(input_file)

    cleaned_output_file = cleaned_output_file or default_cleaned_name
    report_output_file = report_output_file or default_report_name

    df_raw = load_data(input_path)
    cleaned_df, summarized_report_df = prepare_accounting_data(df_raw)

    cleaned_output_path = save_cleaned_data(cleaned_df, cleaned_output_file)
    report_output_path = save_quality_report(summarized_report_df, report_output_file)
    cleaned_csv_output_path = save_csv_copy(cleaned_df, cleaned_output_file)

    safe_print(f"Saved cleaned data to: {cleaned_output_path}")
    safe_print(f"Saved data quality report to: {report_output_path}")
    safe_print(f"Saved cleaned data CSV copy to: {cleaned_csv_output_path}")
    return 0


if __name__ == "__main__":
    input_file_arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_FILE_NAME
    cleaned_output_arg = sys.argv[2] if len(sys.argv) > 2 else None
    report_output_arg = sys.argv[3] if len(sys.argv) > 3 else None
    raise SystemExit(main(input_file_arg, cleaned_output_arg, report_output_arg))



