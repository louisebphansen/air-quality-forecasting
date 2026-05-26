import os
import joblib
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima import ndiffs, nsdiffs


def sarimax_tuning(
    target_col,
    predictor_cols,      # list of str, or [] for univariate
    fit_name,            # e.g. "NO2_fit_2"
    y_train,
    y_test,
    X_train_scaled,      # pre-scaled slice for this fit's columns (or None if univariate)
    X_test_scaled,
    m=24
):
    """
    Grid-searches SARIMA orders for one (pollutant, fit) combination.
    Saves the best model residual plots.

    Returns a spec dict:
        {
            "name":           fit_name,
            "order":          (p, d, q),
            "seasonal_order": (P, D, 1, m),
            "exog_cols":      predictor_cols,
            "aic":            float,
        }
    """
    
    d = ndiffs(y_train, test="kpss")
    D = nsdiffs(y_train, m=m, test="ch")
    print(f"  [{fit_name}] d={d}, D={D}")

    results = []
    for p in range(0, 4):
        for q in range(1, 4):
            for P in range(0, 2):
                try:
                    mdl = SARIMAX(
                        y_train,
                        exog=X_train_scaled,
                        order=(p, d, q),
                        seasonal_order=(P, D, 1, m)
                    ).fit(disp=False)
                    results.append({
                        "order":          (p, d, q),
                        "seasonal_order": (P, D, 1, m),
                        "aic":            mdl.aic,
                        "model":          mdl,
                    })
                    print(f"    SARIMA({p},{d},{q})({P},{D},1)[{m}]  AIC={mdl.aic:.2f}")
                except Exception:
                    pass

    if not results:
        raise RuntimeError(f"All candidates failed for {fit_name}")

    best       = min(results, key=lambda x: x["aic"])
    best_model = best["model"]
    p_b, _, q_b = best["order"]
    print(f"  → Best: order={best['order']}  seasonal={best['seasonal_order']}  AIC={best['aic']:.2f}")
    print(best_model.summary())

    os.makedirs("../out/plots/ARIMA_residuals", exist_ok=True)

    residuals = best_model.resid
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(residuals)
    axes[0, 0].axhline(0, color="red", linestyle="--")
    axes[0, 0].set_title(f"{fit_name} — residuals over time")
    axes[0, 1].hist(residuals, bins=40)
    axes[0, 1].set_title("Residual distribution")
    plot_acf(residuals,  ax=axes[1, 0], lags=48)
    axes[1, 0].set_title("ACF of residuals")
    plot_pacf(residuals, ax=axes[1, 1], lags=48)
    axes[1, 1].set_title("PACF of residuals")
    plt.tight_layout()
    plt.savefig(f"../out/plots/ARIMA_residuals/{fit_name}_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "name":           fit_name,
        "order":          best["order"],
        "seasonal_order": best["seasonal_order"],
        "exog_cols":      predictor_cols,
        "aic":            best["aic"],
        "fitted_model":   best_model,
}
