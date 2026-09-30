import pandas as pd
import numpy as np

def auto_detect_datetime(df):
    """
    Advanced automatic datetime detection and conversion.
    Handles:
    - mixed date formats
    - stock market dates
    - slash/dash formats
    - invalid parsing safely
    """
    datetime_report = []

    date_keywords = [
        'date', 'time', 'timestamp',
        'year', 'month', 'day'
    ]

    for col in df.columns:
        original_dtype = df[col].dtype

        # check object columns OR columns with date keywords
        if(original_dtype == 'object' or any(keyword in col.lower() for keyword in date_keywords)):

            # Clean values
            cleaned_col = (df[col].astype(str).str.strip().replace(['nan', 'None', ''], np.nan))

            # ---------- TRY MULTIPLE DATE PARSING STRATEGIES ----------

            converted_col = pd.to_datetime(
                cleaned_col,
                errors='coerce',
                dayfirst=True
            )


            success_ratio = converted_col.notnull().mean()
            failed_count = converted_col.isnull().sum()

            # ---------- SECOND ATTEMPT ----------
            if success_ratio < 0.80:

                converted_col_alt = pd.to_datetime(
                    cleaned_col,
                    errors='coerce',
                    dayfirst=False
                )

                alt_ratio = converted_col_alt.notnull().mean()

                # Use better conversion
                if alt_ratio > success_ratio:
                    converted_col = converted_col_alt
                    success_ratio = alt_ratio
                    failed_count = converted_col.isnull().sum()

            # ---------- FINAL DECISION ----------

            if success_ratio >= 0.70:

                df[col] = converted_col

                detection_status = "Converted to Datetime"

            else:

                detection_status = "Kept as Object"

            datetime_report.append({
                "Column": col,
                "Conversion Success %": round(success_ratio * 100, 2),
                "Failed Rows": failed_count,
                "Status": detection_status
            })

    report_df = pd.DataFrame(datetime_report)

    return df, report_df