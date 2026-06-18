```
from scipy.stats import theilslopes

def calc_step_metrics(x, d, eps=1e-8, edge_frac=0.2):
    slope, intercept, _, _ = theilslopes(d, x)   # robust 기준 트렌드
    trend = slope * x + intercept
    err = d - trend
    n = len(err)

    err_diff = np.diff(err)
    mad = np.median(np.abs(err_diff - np.median(err_diff)))
    sigma_robust = max(mad * 1.4826 / np.sqrt(2), np.std(err) * 0.05, eps)

    cusum = np.cumsum(err - np.mean(err))
    cusum_stat = np.max(np.abs(cusum)) / (sigma_robust * np.sqrt(n))

    edge = max(int(n * edge_frac), 3)
    head = np.median(err[:edge])
    tail = np.median(err[-edge:])
    persistence_ratio = np.abs(tail - head) / sigma_robust

    step_score = min(cusum_stat, persistence_ratio)
    return {"step_score": step_score}

```

```
def calc_fit_metrics(x, d, rul, eps=1e-8, edge_frac=0.2):
    err = rul - d
    n = len(err)

    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    bias = np.mean(err)
    max_err = np.max(np.abs(err))

    if np.std(d) < eps or np.std(rul) < eps:
        corr = np.nan
    else:
        corr = np.corrcoef(d, rul)[0, 1]

    ss_res = np.sum((d - rul) ** 2)
    ss_tot = np.sum((d - np.mean(d)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > eps else np.nan

    err_diff = np.diff(err)
    mad = np.median(np.abs(err_diff - np.median(err_diff)))
    sigma_robust = max(mad * 1.4826 / np.sqrt(2), np.std(err) * 0.05, eps)

    cusum = np.cumsum(err - np.mean(err))
    cusum_stat = np.max(np.abs(cusum)) / (sigma_robust * np.sqrt(n))

    edge = max(int(n * edge_frac), 3)
    head_level = np.median(err[:edge])
    tail_level = np.median(err[-edge:])
    persistence_ratio = np.abs(tail_level - head_level) / sigma_robust

    step_score = min(cusum_stat, persistence_ratio)

    return {
        "rmse": rmse, "mae": mae, "bias": bias, "max_error": max_err,
        "corr": corr, "r2": r2,
        "step_score": step_score,
    }

```


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


```
def add_weighted_violation_score(df, weights, violations_col="violations"):
    """
    df["violations"]에 들어있는 metric 리스트를 기준으로
    metric별 weight를 합산하여 score_violations 컬럼 추가.
    """

    def calc_score(vs):
        if isinstance(vs, str):
            # CSV에서 읽어서 "['rmse', 'mae']" 같은 문자열인 경우 처리
            import ast
            try:
                vs = ast.literal_eval(vs)
            except Exception:
                vs = [vs]

        if vs is None:
            return 0.0

        return sum(weights.get(v, 1.0) for v in vs)

    df = df.copy()
    df["score_violations"] = df[violations_col].apply(calc_score)

    return df


weights = {
    "rmse": 3.0,
    "mae": 2.0,
    "bias": 2.0,
    "max_error": 2.5,
    "corr": 1.5,
    "r2": 1.5,
    "grad_std": 1.0,
    "curvature_noise": 3.0,
    "diff_std": 2.0,
}

df = add_weighted_violation_score(df, weights)

df_sorted = df.sort_values(
    by="score_violations",
    ascending=False
).reset_index(drop=True)

df_sorted[
    ["file", "n_violations", "score_violations", "violations"]
].head(20)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.hist(df["score_violations"], bins=20)
plt.xlabel("score_violations")
plt.ylabel("count")
plt.title("Weighted Violation Score Distribution")
plt.grid(True)
plt.show()
```



```
import ast
import pandas as pd

def add_weighted_violation_score(
    df,
    weights,
    violations_col="violations",
    grad_std_threshold=5.0,
):
    """
    df["violations"]에 들어있는 metric 리스트를 기준으로
    metric별 weight를 합산하여 score_violations 컬럼 추가.

    단, grad_std는 실제 row["grad_std"] 값이 grad_std_threshold보다 클 때만
    weight를 반영.
    """

    def parse_violations(vs):
        if isinstance(vs, str):
            try:
                vs = ast.literal_eval(vs)
            except Exception:
                vs = [vs]

        if vs is None:
            return []

        if isinstance(vs, float) and pd.isna(vs):
            return []

        return vs

    def calc_score(row):
        vs = parse_violations(row[violations_col])

        score = 0.0

        for v in vs:
            if v == "grad_std":
                if row["grad_std"] > grad_std_threshold:
                    score += weights.get(v, 1.0)
            else:
                score += weights.get(v, 1.0)

        return score

    df = df.copy()

    df["score_violations"] = df.apply(
        calc_score,
        axis=1
    )

    return df

weights = {
    "rmse": 3.0,
    "mae": 2.0,
    "bias": 2.0,
    "max_error": 2.5,
    "corr": 1.5,
    "r2": 1.5,
    "grad_std": 1.0,
    "curvature_noise": 3.0,
    "diff_std": 2.0,
}

df = add_weighted_violation_score(
    df,
    weights,
    grad_std_threshold=5.0
)
```

