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