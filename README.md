### Miscs

d 데이터 업데이트
→ rul 예측값 업데이트
→ rul vs d 정합성 평가
→ PASS면 고객 화면 노출
→ FAIL이면 숨김 또는 “데이터 안정화 중” 처리

```
import numpy as np
from scipy.interpolate import interp1d

def align_curves(dx, dy, rul_x, rul_y, n=1000):
    xmin = max(np.min(dx), np.min(rul_x))
    xmax = min(np.max(dx), np.max(rul_x))

    x_common = np.linspace(xmin, xmax, n)

    d_func = interp1d(dx, dy, kind="linear", bounds_error=False, fill_value="extrapolate")
    rul_func = interp1d(rul_x, rul_y, kind="linear", bounds_error=False, fill_value="extrapolate")

    d_common = d_func(x_common)
    rul_common = rul_func(x_common)

    return x_common, d_common, rul_common
```

```
def calc_fit_metrics(x, d, rul):
    err = rul - d

    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    bias = np.mean(err)
    max_err = np.max(np.abs(err))

    corr = np.corrcoef(d, rul)[0, 1]

    ss_res = np.sum((d - rul)**2)
    ss_tot = np.sum((d - np.mean(d))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "max_error": max_err,
        "corr": corr,
        "r2": r2,
    }
```

```
RMSE / MAE 낮음     → 전체적으로 잘 맞음
Bias 양수           → rul이 d보다 지속적으로 높게 예측
Bias 음수           → rul이 d보다 지속적으로 낮게 예측
Max error 큼        → 특정 구간에서 크게 이탈
Corr 높음           → 모양은 비슷함
R² 낮음             → 설명력이 약함


def calc_noise_metrics(x, d):
    # 1차 기울기
    grad = np.gradient(d, x)

    # 2차 변화량: 곡선의 울렁임/진동성
    curvature = np.gradient(grad, x)

    noise_score = np.std(curvature)
    grad_std = np.std(grad)

    # 인접 포인트 변화량
    diff_std = np.std(np.diff(d))

    return {
        "grad_std": grad_std,
        "curvature_noise": noise_score,
        "diff_std": diff_std,
    }

curvature_noise 큼
→ d가 매끄럽지 않고 출렁임이 많음

diff_std 큼
→ 인접 데이터 간 점프가 큼

grad_std 큼
→ 열화 속도 변화가 불안정함




def calc_gradient_metrics(x, d, rul):
    d_grad = np.gradient(d, x)
    rul_grad = np.gradient(rul, x)

    grad_err = rul_grad - d_grad

    grad_rmse = np.sqrt(np.mean(grad_err**2))
    grad_corr = np.corrcoef(d_grad, rul_grad)[0, 1]

    return {
        "grad_rmse": grad_rmse,
        "grad_corr": grad_corr,
    }
현재 값은 맞아 보여도
앞으로의 degradation trend를 잘못 따라가고 있을 가능성



def judge_display_quality(fit, noise, grad):
    reasons = []

    if fit["rmse"] > 0.03:
        reasons.append("RMSE too high")

    if abs(fit["bias"]) > 0.02:
        reasons.append("Systematic bias too high")

    if fit["corr"] < 0.90:
        reasons.append("Correlation too low")

    if fit["r2"] < 0.80:
        reasons.append("R2 too low")

    if grad["grad_corr"] < 0.70:
        reasons.append("Gradient trend mismatch")

    if noise["curvature_noise"] > 0.01:
        reasons.append("Measured data too noisy")

    display_ok = len(reasons) == 0

    return display_ok, reasons







import matplotlib.pyplot as plt

def plot_rul_quality(x, d, rul):
    err = rul - d

    d_grad = np.gradient(d, x)
    rul_grad = np.gradient(rul, x)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Overlay
    axes[0, 0].plot(x, d, label="d")
    axes[0, 0].plot(x, rul, label="rul")
    axes[0, 0].set_title("d vs rul")
    axes[0, 0].legend()
    axes[0, 0].grid()

    # 2. Residual
    axes[0, 1].plot(x, err)
    axes[0, 1].axhline(0, linestyle="--")
    axes[0, 1].set_title("Residual: rul - d")
    axes[0, 1].grid()

    # 3. Predicted vs Actual
    axes[1, 0].scatter(d, rul, s=10)

    mn = min(np.min(d), np.min(rul))
    mx = max(np.max(d), np.max(rul))
    axes[1, 0].plot([mn, mx], [mn, mx], linestyle="--")

    axes[1, 0].set_xlabel("d")
    axes[1, 0].set_ylabel("rul")
    axes[1, 0].set_title("Predicted vs Actual")
    axes[1, 0].grid()

    # 4. Gradient
    axes[1, 1].plot(x, d_grad, label="d gradient")
    axes[1, 1].plot(x, rul_grad, label="rul gradient")
    axes[1, 1].set_title("Gradient comparison")
    axes[1, 1].legend()
    axes[1, 1].grid()

    plt.tight_layout()
    plt.show()




x, d_common, rul_common = align_curves(dx, dy, rul_x, rul_y)

fit = calc_fit_metrics(x, d_common, rul_common)
noise = calc_noise_metrics(x, d_common)
grad = calc_gradient_metrics(x, d_common, rul_common)

display_ok, reasons = judge_display_quality(fit, noise, grad)

print("display_ok:", display_ok)
print("reasons:", reasons)

print("fit:", fit)
print("noise:", noise)
print("gradient:", grad)

plot_rul_quality(x, d_common, rul_common)





PASS
→ 고객에게 rul 표시

WARNING
→ 내부 모니터링 대상
→ 고객에게는 표시 가능하되 confidence 낮게 처리

FAIL_MODEL
→ rul과 d가 안 맞음
→ 고객에게 rul 미표시

FAIL_DATA
→ d 자체가 너무 noisy
→ 고객에게 “데이터 안정화 중” 또는 “추가 데이터 필요” 처리


status = "PASS"        # 정상 표시
status = "WARNING"     # 표시 가능하지만 주의
status = "FAIL_MODEL"  # 모델 정합성 부족
status = "FAIL_DATA"   # 입력 데이터 품질 부족

```