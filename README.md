```
import numpy as np
import pandas as pd


def get_iqr_bounds(s):
    s = pd.to_numeric(s, errors="coerce").dropna()

    q1 = s.quantile(0.25)
    q2 = s.quantile(0.50)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return q1, q2, q3, iqr, lower, upper


def extract_metric_tolerances(df):
    upper_metrics = [
        "rmse",
        "mae",
        "max_error",
        "grad_std",
        "curvature_noise",
        "diff_std",
    ]

    lower_metrics = [
        "corr",
        "r2",
    ]

    two_side_metrics = [
        "bias",
    ]

    rows = []

    for col in upper_metrics:
        q1, q2, q3, iqr, lower, upper = get_iqr_bounds(df[col])

        rows.append({
            "metric": col,
            "direction": "upper",
            "tolerance_lower": np.nan,
            "tolerance_upper": upper,
            "q1": q1,
            "median": q2,
            "q3": q3,
            "iqr": iqr,
            "p05": df[col].quantile(0.05),
            "p95": df[col].quantile(0.95),
        })

    for col in lower_metrics:
        q1, q2, q3, iqr, lower, upper = get_iqr_bounds(df[col])

        rows.append({
            "metric": col,
            "direction": "lower",
            "tolerance_lower": lower,
            "tolerance_upper": np.nan,
            "q1": q1,
            "median": q2,
            "q3": q3,
            "iqr": iqr,
            "p05": df[col].quantile(0.05),
            "p95": df[col].quantile(0.95),
        })

    for col in two_side_metrics:
        q1, q2, q3, iqr, lower, upper = get_iqr_bounds(df[col])

        rows.append({
            "metric": col,
            "direction": "two_side",
            "tolerance_lower": lower,
            "tolerance_upper": upper,
            "q1": q1,
            "median": q2,
            "q3": q3,
            "iqr": iqr,
            "p05": df[col].quantile(0.05),
            "p95": df[col].quantile(0.95),
        })

    tol_df = pd.DataFrame(rows)

    return tol_df



tol_df = extract_metric_tolerances(df)
print(tol_df)
tol_df.to_csv("metric_tolerances.csv", index=False)





def check_metric_tolerance(row, tol_df):
    violations = []

    for _, t in tol_df.iterrows():
        metric = t["metric"]
        direction = t["direction"]
        value = row[metric]

        if direction == "upper":
            if value > t["tolerance_upper"]:
                violations.append(metric)

        elif direction == "lower":
            if value < t["tolerance_lower"]:
                violations.append(metric)

        elif direction == "two_side":
            if value < t["tolerance_lower"] or value > t["tolerance_upper"]:
                violations.append(metric)

    return violations



df["violations"] = df.apply(
    lambda row: check_metric_tolerance(row, tol_df),
    axis=1
)

df["n_violations"] = df["violations"].apply(len)
```