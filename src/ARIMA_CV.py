import os
import joblib
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from pmdarima.model_selection import RollingForecastCV
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def compare_sarimax_models(
    pollutant,
    specs,
    y_train,
    y_test,
    X_train_scaled,
    X_test_scaled,
    all_predictor_cols,
    m=24,
    forecast_horizon=96,
    step=96,
    initial=8280,
    maxiter=200,
    n_boot=500,
    boot_alpha=0.05,
):
    print("Initialising")
    rolling_cv = RollingForecastCV(h=forecast_horizon, step=step, initial=initial)
    n_actual   = sum(1 for _ in rolling_cv.split(y_train))
    print(f"  CV: {n_actual} folds  (step={step}, h={forecast_horizon})")

    performance = []
    cv_rows     = []

    # ── Step 1: fit once per spec, CV over folds ──────────────────────────
    for spec in specs:
        print(f"Running {spec['name']} now")
        exog_cols = spec["exog_cols"]
        if exog_cols:
            col_idx   = [all_predictor_cols.index(c) for c in exog_cols]
            X_tr_spec = X_train_scaled[:, col_idx]
        else:
            X_tr_spec = None

        mdl = spec.get("fitted_model") or SARIMAX(
            y_train,
            exog=X_tr_spec,
            order=spec["order"],
            seasonal_order=spec["seasonal_order"]
        ).fit(disp=False, maxiter=maxiter)

        window_rmses = []
        for fold, (_, val_idx) in enumerate(rolling_cv.split(y_train)):
            y_val_f = y_train[val_idx]
            X_val_f = X_tr_spec[val_idx] if X_tr_spec is not None else None
            try:
                preds       = mdl.forecast(steps=len(y_val_f), exog=X_val_f)
                window_rmse = np.sqrt(np.mean((preds - y_val_f) ** 2))
                window_rmses.append(window_rmse)
                print(f"    fold {fold+1}/{n_actual}  RMSE={window_rmse:.4f}")

                cv_rows.append({
                    "spec_name":      spec["name"],
                    "order":          str(spec["order"]),
                    "seasonal_order": str(spec["seasonal_order"]),
                    "exog_cols":      str(exog_cols),
                    "fold":           fold + 1,
                    "val_start_idx":  val_idx[0],
                    "val_end_idx":    val_idx[-1],
                    "n_obs":          len(y_val_f),
                    "rmse":           window_rmse,
                })
            except Exception as e:
                window_rmses.append(np.inf)
                cv_rows.append({
                    "spec_name": spec["name"], "fold": fold + 1,
                    "rmse": np.inf, "error": str(e),
                })

        mean_cv_rmse = np.mean(window_rmses)
        # ── SD across folds — excludes inf (failed folds) ─────────────────
        finite_rmses = [r for r in window_rmses if np.isfinite(r)]
        sd_cv_rmse   = np.std(finite_rmses) if len(finite_rmses) > 1 else np.nan

        performance.append({
            "name":        spec["name"],
            "spec":        spec,
            "cv_rmse":     mean_cv_rmse,
            "cv_rmse_sd":  sd_cv_rmse,
        })
        print(f"  → {spec['name']:25s}  mean CV RMSE={mean_cv_rmse:.4f}  SD={sd_cv_rmse:.4f}\n")

    # ── Save CV results to CSV ─────────────────────────────────────────────
    cv_df = pd.DataFrame(cv_rows)

    # ── Append per-spec summary rows (mean + SD) ──────────────────────────
    summary_rows = pd.DataFrame([{
        "spec_name": p["name"],
        "order":     str(p["spec"]["order"]),
        "seasonal_order": str(p["spec"]["seasonal_order"]),
        "fold":      "MEAN",
        "rmse":      p["cv_rmse"],
        "rmse_sd":   p["cv_rmse_sd"],
    } for p in performance])
    cv_df = pd.concat([cv_df, summary_rows], ignore_index=True)

    os.makedirs("../out/cv_results/ARIMA", exist_ok=True)
    cv_df.to_csv(f"../out/cv_results/ARIMA/cv_results_{pollutant}.csv", index=False)
    print(f"  → CV results saved to ../out/cv_results/ARIMA/cv_results_{pollutant}.csv")    

    # ── Step 2: pick best by mean CV RMSE ────────────────────────────────
    best_result = min(performance, key=lambda x: x["cv_rmse"])
    best_spec   = best_result["spec"]
    print(f"\n  Winner: {best_result['name']}  "
          f"CV RMSE={best_result['cv_rmse']:.4f}  SD={best_result['cv_rmse_sd']:.4f}")

    if best_spec["exog_cols"]:
        col_idx   = [all_predictor_cols.index(c) for c in best_spec["exog_cols"]]
        X_tr_best = X_train_scaled[:, col_idx]
        X_te_best = X_test_scaled[:,  col_idx]
    else:
        X_tr_best = None
        X_te_best = None

    # ── Step 3: refit winner on full training data ────────────────────────
    final_model = SARIMAX(
        y_train,
        exog=X_tr_best,
        order=best_spec["order"],
        seasonal_order=best_spec["seasonal_order"]
    ).fit(disp=False, maxiter=maxiter)

    # ── Step 4: holdout point forecast + parametric uncertainty ──────────
    forecast_obj  = final_model.get_forecast(steps=len(y_test), exog=X_te_best)
    summary_frame = forecast_obj.summary_frame(alpha=boot_alpha)
    holdout_fc    = summary_frame["mean"].values

    holdout_mse  = np.mean((holdout_fc - y_test) ** 2)
    holdout_rmse = np.sqrt(holdout_mse)
    holdout_smape = np.mean(
        2 * np.abs(holdout_fc - y_test) / (np.abs(holdout_fc) + np.abs(y_test) + 1e-8)
    ) * 100
    print(f"  Holdout RMSE={holdout_rmse:.4f}  SMAPE={holdout_smape:.2f}%")

    # ── Step 5: bootstrap uncertainty on holdout ─────────────────────────
    residuals  = final_model.resid
    rng        = np.random.default_rng(seed=42)
    boot_paths = np.zeros((n_boot, len(y_test)))

    for b in range(n_boot):
        noise         = rng.choice(residuals, size=len(y_test), replace=True)
        boot_paths[b] = holdout_fc + noise

    lower_pi       = np.percentile(boot_paths, 100 * (boot_alpha / 2),     axis=0)
    upper_pi       = np.percentile(boot_paths, 100 * (1 - boot_alpha / 2), axis=0)
    interval_width = upper_pi - lower_pi

    coverage = np.mean((y_test >= lower_pi) & (y_test <= upper_pi))
    print(f"  Bootstrap {int((1-boot_alpha)*100)}% PI  coverage={coverage:.3f}  "
          f"mean width={interval_width.mean():.4f}")

    # ── Save holdout forecast + uncertainty ───────────────────────────────
    holdout_df = pd.DataFrame({
        "step":             np.arange(1, len(y_test) + 1),
        "actual":           y_test,
        "forecast":         holdout_fc,
        "parametric_se":    summary_frame["mean_se"].values,
        "parametric_lower": summary_frame["mean_ci_lower"].values,
        "parametric_upper": summary_frame["mean_ci_upper"].values,
        "bootstrap_lower":  lower_pi,
        "bootstrap_upper":  upper_pi,
        "bootstrap_width":  interval_width,
    })

    os.makedirs("../out/forecasts/ARIMA", exist_ok=True)
    holdout_df.to_csv(f"../out/forecasts/ARIMA/holdout_forecast_{pollutant}.csv", index=False)
    print(f"  → Holdout forecast saved to ../out/forecasts/ARIMA/holdout_forecast_{pollutant}.csv")

    # ── Save holdout summary metrics ──────────────────────────────────────
    pd.DataFrame([{
        "pollutant":        pollutant,
        "best_spec":        best_result["name"],
        "order":            str(best_spec["order"]),
        "seasonal_order":   str(best_spec["seasonal_order"]),
        "cv_rmse":          best_result["cv_rmse"],
        "cv_rmse_sd":       best_result["cv_rmse_sd"],
        "holdout_rmse":     holdout_rmse,
        "holdout_smape":    holdout_smape,
        "boot_coverage":    coverage,
        "boot_mean_width":  interval_width.mean(),
    }]).to_csv(f"../out/forecasts/ARIMA/holdout_summary_{pollutant}.csv", index=False)
    print(f"  → Holdout summary saved to ../out/forecasts/ARIMA/holdout_summary_{pollutant}.csv")

    os.makedirs("../out/modelfits/ARIMA", exist_ok=True)
    save_path = f"../out/modelfits/ARIMA/{best_spec['name'].split('_')[0]}_BEST_{best_spec['name']}.pkl"
    joblib.dump(final_model, save_path)
    print(f"  → Model saved to {save_path}")