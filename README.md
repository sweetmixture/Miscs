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


```
df_sorted = df.sort_values(
    by="n_violations",
    ascending=False
).reset_index(drop=True)


import pandas as pd
import matplotlib.pyplot as plt


def plot_file_curve(file_path, title=None):
    fdf = pd.read_csv(file_path)

    plt.figure(figsize=(8, 5))

    plt.plot(
        fdf["fieldf_x"],
        fdf["fieldf_y"],
        marker="o",
        linestyle="-",
        label="data"
    )

    plt.plot(
        fdf["cmodel_x"],
        fdf["cmodel_y"],
        linestyle="--",
        label="rul"
    )

    if title is None:
        title = file_path

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



for _, row in df_sorted.head(10).iterrows():
    file_path = row["file"]
    n_v = row["n_violations"]

    title = f"{file_path} | n_violations={n_v}"

    plot_file_curve(file_path, title=title)



target_n = 3

df_target = df[df["n_violations"] == target_n]

for _, row in df_target.iterrows():
    file_path = row["file"]

    title = f"{file_path} | n_violations={target_n}"

    plot_file_curve(file_path, title=title)




def plot_by_violation_group(df, max_per_group=5):
    df_sorted = df.sort_values(
        by="n_violations",
        ascending=False
    )

    for n_v, g in df_sorted.groupby("n_violations", sort=False):
        print(f"\n===== n_violations = {n_v} =====")

        for _, row in g.head(max_per_group).iterrows():
            file_path = row["file"]

            title = f"{file_path} | n_violations={n_v}"

            plot_file_curve(file_path, title=title)

plot_by_violation_group(df, max_per_group=3)





def plot_from_summary_row(row):
    file_path = row["file"]

    title = (
        f"{file_path}\n"
        f"n_violations={row['n_violations']} | "
        f"rmse={row['rmse']:.4g}, "
        f"mae={row['mae']:.4g}, "
        f"corr={row['corr']:.4g}, "
        f"r2={row['r2']:.4g}"
    )

    plot_file_curve(file_path, title=title)

for _, row in df_sorted.head(10).iterrows():
    plot_from_summary_row(row)
```


