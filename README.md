```

def calc_gradient_metrics(x, d, rul, eps=1e-8):
    d_grad = np.gradient(d, x)
    rul_grad = np.gradient(rul, x)

    grad_err = rul_grad - d_grad
    grad_rmse = np.sqrt(np.mean(grad_err**2))

    if np.std(d_grad) < eps or np.std(rul_grad) < eps:
        grad_corr = np.nan
    else:
        grad_corr = np.corrcoef(d_grad, rul_grad)[0, 1]

    d_sign = np.sign(d_grad)
    rul_sign = np.sign(rul_grad)

    d_sign[np.abs(d_grad) < eps] = 0
    rul_sign[np.abs(rul_grad) < eps] = 0

    valid = (d_sign != 0) & (rul_sign != 0)

    slope_sign_agreement = (
        np.mean(d_sign[valid] == rul_sign[valid])
        if np.sum(valid) > 0
        else np.nan
    )

    d_mean_slope = np.mean(d_grad)
    rul_mean_slope = np.mean(rul_grad)

    same_global_direction = (
        np.sign(d_mean_slope) == np.sign(rul_mean_slope)
        if abs(d_mean_slope) > eps and abs(rul_mean_slope) > eps
        else False
    )

    return {
        "grad_rmse": grad_rmse,
        "grad_corr": grad_corr,
        "slope_sign_agreement": slope_sign_agreement,
        "same_global_direction": bool(same_global_direction),
        "d_mean_slope": d_mean_slope,
        "rul_mean_slope": rul_mean_slope,
    }





def calc_fit_metrics(x, d, rul, eps=1e-8):
    err = rul - d

    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    bias = np.mean(err)
    max_err = np.max(np.abs(err))

    if np.std(d) < eps or np.std(rul) < eps:
        corr = np.nan
    else:
        corr = np.corrcoef(d, rul)[0, 1]

    ss_res = np.sum((d - rul)**2)
    ss_tot = np.sum((d - np.mean(d))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > eps else np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "max_error": max_err,
        "corr": corr,
        "r2": r2,
    }



def judge_display_quality(fit, noise, grad):
    reasons = []

    if not grad["same_global_direction"]:
        reasons.append("Opposite global trend direction")

    if (
        not np.isnan(grad["slope_sign_agreement"])
        and grad["slope_sign_agreement"] < 0.8
    ):
        reasons.append("Local slope direction mismatch")

    if fit["rmse"] > 0.03:
        reasons.append("RMSE too high")

    if abs(fit["bias"]) > 0.02:
        reasons.append("Systematic bias too high")

    if not np.isnan(fit["corr"]) and fit["corr"] < 0.90:
        reasons.append("Correlation too low")

    if not np.isnan(fit["r2"]) and fit["r2"] < 0.80:
        reasons.append("R2 too low")

    if not np.isnan(grad["grad_corr"]) and grad["grad_corr"] < 0.70:
        reasons.append("Gradient correlation too low")

    if noise["curvature_noise"] > 0.01:
        reasons.append("Measured data too noisy")

    display_ok = len(reasons) == 0

    return display_ok, reasons


1. 전체 방향이 같은가?
2. 국소 기울기 방향도 같은가?
3. 값 자체가 맞는가?
4. gradient trend가 맞는가?
5. d 자체가 noisy한가?

```

```

def penalty_score(value, good, bad):
    """
    value <= good 이면 100점
    value >= bad 이면 0점
    그 사이는 선형 감소
    """
    if np.isnan(value):
        return 50.0

    if value <= good:
        return 100.0

    if value >= bad:
        return 0.0

    return 100.0 * (bad - value) / (bad - good)


def reward_score(value, bad, good):
    """
    value >= good 이면 100점
    value <= bad 이면 0점
    그 사이는 선형 증가
    """
    if np.isnan(value):
        return 50.0

    if value >= good:
        return 100.0

    if value <= bad:
        return 0.0

    return 100.0 * (value - bad) / (good - bad)


def judge_display_quality_score(fit, noise, grad):
    reasons = []

    # -------------------------
    # 1. Data quality score
    # -------------------------
    data_scores = {}

    data_scores["curvature_noise"] = penalty_score(
        noise["curvature_noise"],
        good=0.003,
        bad=0.015,
    )

    data_scores["diff_std"] = penalty_score(
        noise["diff_std"],
        good=0.003,
        bad=0.015,
    )

    data_scores["grad_std"] = penalty_score(
        noise["grad_std"],
        good=0.003,
        bad=0.015,
    )

    data_score = np.mean(list(data_scores.values()))

    # -------------------------
    # 2. Model fit score
    # -------------------------
    model_scores = {}

    model_scores["rmse"] = penalty_score(
        fit["rmse"],
        good=0.01,
        bad=0.05,
    )

    model_scores["mae"] = penalty_score(
        fit["mae"],
        good=0.008,
        bad=0.04,
    )

    model_scores["bias"] = penalty_score(
        abs(fit["bias"]),
        good=0.005,
        bad=0.03,
    )

    model_scores["corr"] = reward_score(
        fit["corr"],
        bad=0.70,
        good=0.95,
    )

    model_scores["r2"] = reward_score(
        fit["r2"],
        bad=0.50,
        good=0.90,
    )

    model_scores["grad_corr"] = reward_score(
        grad["grad_corr"],
        bad=0.50,
        good=0.85,
    )

    model_scores["slope_sign_agreement"] = reward_score(
        grad["slope_sign_agreement"],
        bad=0.70,
        good=0.95,
    )

    model_score = np.mean(list(model_scores.values()))

    # -------------------------
    # 3. Hard fail conditions
    # -------------------------
    hard_fail_data = False
    hard_fail_model = False

    if noise["curvature_noise"] > 0.02:
        hard_fail_data = True
        reasons.append("Measured data is too noisy")

    if noise["diff_std"] > 0.02:
        hard_fail_data = True
        reasons.append("Measured data has excessive point-to-point jumps")

    if not grad["same_global_direction"]:
        hard_fail_model = True
        reasons.append("Opposite global trend direction")

    if (
        not np.isnan(grad["slope_sign_agreement"])
        and grad["slope_sign_agreement"] < 0.6
    ):
        hard_fail_model = True
        reasons.append("Severe local slope direction mismatch")

    if fit["rmse"] > 0.07:
        hard_fail_model = True
        reasons.append("RMSE is critically high")

    # -------------------------
    # 4. Total score
    # -------------------------
    total_score = 0.4 * data_score + 0.6 * model_score

    # -------------------------
    # 5. Status decision
    # -------------------------
    if hard_fail_data and hard_fail_model:
        status = "FAIL_BOTH"

    elif hard_fail_data:
        status = "FAIL_DATA"

    elif hard_fail_model:
        status = "FAIL_MODEL"

    elif data_score < 50 and model_score < 50:
        status = "FAIL_BOTH"

    elif data_score < 50:
        status = "FAIL_DATA"

    elif model_score < 50:
        status = "FAIL_MODEL"

    elif total_score < 70:
        status = "WARNING"

    else:
        status = "PASS"

    return {
        "status": status,
        "display_ok": status == "PASS",
        "total_score": total_score,
        "data_score": data_score,
        "model_score": model_score,
        "data_scores": data_scores,
        "model_scores": model_scores,
        "reasons": reasons,
    }




fit = calc_fit_metrics(x, d_common, rul_common)
noise = calc_noise_metrics(x, d_common)
grad = calc_gradient_metrics(x, d_common, rul_common)

quality = judge_display_quality_score(fit, noise, grad)

print(quality["status"])
print(quality["total_score"])
print(quality["data_score"])
print(quality["model_score"])
print(quality["reasons"])






PASS
→ 고객에게 rul 표시

WARNING
→ 표시 가능하나 confidence 낮음
→ 내부 모니터링 대상

FAIL_DATA
→ d 자체가 noisy
→ 모델 문제가 아니라 입력 데이터 품질 문제

FAIL_MODEL
→ d는 괜찮은데 rul이 d를 못 따라감

FAIL_BOTH
→ d도 noisy하고 rul도 안 맞음



quality = {
    "status": "FAIL_MODEL",
    "display_ok": False,
    "total_score": 42.3,
    "data_score": 81.5,
    "model_score": 16.2,
    "reasons": [
        "Opposite global trend direction",
        "RMSE is critically high"
    ]
}

```