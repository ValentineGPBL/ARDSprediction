import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats



BUCKETS = [
    ("severe",   0,   100),
    ("moderate", 100, 200),
    ("mild",     200, 300),
    ("normal",   300, float("inf")),
]


def pf_bucket(pf):
    for name, lo, hi in BUCKETS:
        if lo <= pf < hi:
            return name
    return "normal"


class Evaluator:

    #evaluator=Evaluator(pred_df) placeholder.csv as a dataframe
    #evaluator.run() prints all metrics + saves plots


    def __init__(self, pred_df: pd.DataFrame, n_bootstrap: int = 1000, output_dir: str = "."):

        
        self.df          = pred_df.dropna(subset=["true_pf", "pf_mean"]).reset_index(drop=True)
        self.n_bootstrap = n_bootstrap
        self.output_dir  = output_dir

        self.true        = self.df["true_pf"].values
        self.pred        = self.df["pf_mean"].values
        self.std         = self.df["pf_std"].values
        self.log_pf_std  = self.df["log_pf_std"].values   # sigma of log(PF)
        self.log_pred    = np.log(np.clip(self.pred, 1e-3, None))
        self.ci_lo = self.df["pf_ci_low"].values
        self.ci_hi = self.df["pf_ci_high"].values

        self.df["bucket"] = self.df["true_pf"].apply(pf_bucket)

    #standrad metrics

    def mae(self, true, pred):
        return np.mean(np.abs(true - pred))

    def medae(self, true, pred):
        return np.median(np.abs(true - pred))

    def rmse(self, true, pred):
        return np.sqrt(np.mean((true - pred) ** 2))


    def bootstrap_ci(self, true, pred, metric_fn, n=None, ci=0.95):

        n = n or self.n_bootstrap
        scores = []
        idx = np.arange(len(true))
        for _ in range(n):
            sample = np.random.choice(idx, size=len(idx), replace=True)
            scores.append(metric_fn(true[sample], pred[sample]))
        lo = np.percentile(scores, (1 - ci) / 2 * 100)
        hi = np.percentile(scores, (1 + ci) / 2 * 100)
        return metric_fn(true, pred), lo, hi



    def regression_metrics(self):
        print("\n Regression metrics (overall)")
        for name, fn in [("MAE", self.mae), ("MedAE", self.medae), ("RMSE", self.rmse)]:
            est, lo, hi = self.bootstrap_ci(self.true, self.pred, fn)
            print(f"  {name:6s} = {est:6.1f} mmHg  (95% CI: {lo:.1f} – {hi:.1f})")


    def bucket_metrics(self):
        print("\n Per-severity bucket breakdown")
        print(f"  {'Bucket':12s}  {'N':>5}  {'MAE':>8}  {'MedAE':>8}  {'RMSE':>8}")
        print(f"  {'-'*12}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}")
        for name, lo_pf, hi_pf in BUCKETS:
            mask = (self.true >= lo_pf) & (self.true < hi_pf)
            n = mask.sum()
            if n < 5:
                print(f"  {name:12s}  {n:>5}  {'(too few)':>8}")
                continue
            t, p = self.true[mask], self.pred[mask]
            mae   = self.mae(t, p)
            medae = self.medae(t, p)
            rmse  = self.rmse(t, p)
            print(f"  {name:12s}  {n:>5}  {mae:>8.1f}  {medae:>8.1f}  {rmse:>8.1f}")

    def coverage(self):

        print("\n Uncertainty calibration - coverage test")
        print(f"  {'Level':>8}  {'Expected':>10}  {'Observed':>10}  {'Status':>10}")
        print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")

        levels = [0.50, 0.80, 0.90, 0.95]
        z_map  = {0.50: 0.674, 0.80: 1.282, 0.90: 1.645, 0.95: 1.960}

        log_true = np.log(np.clip(self.true, 1e-3, None))

        for level in levels:
            z    = z_map[level]
            # log-normal CI: exp(log_mu ± z * sigma_log)
            lo   = np.exp(self.log_pred - z * self.log_pf_std)
            hi   = np.exp(self.log_pred + z * self.log_pf_std)
            obs  = np.mean((self.true >= lo) & (self.true <= hi))
            diff = obs - level
            status = "✓ ok" if abs(diff) < 0.05 else ("↑ overconfident" if diff < 0 else "↓ underconfident")
            print(f"  {level*100:>7.0f}%  {level*100:>9.1f}%  {obs*100:>9.1f}%  {status:>10}")

    def spiegelhalter_test(self):

        print("\n Spiegelhalter z-test (calibration, log-space)")
        log_true = np.log(np.clip(self.true, 1e-3, None))
        var   = self.log_pf_std ** 2          # variance of log(PF)
        z_i   = (log_true - self.log_pred) ** 2 - var
        z_num = z_i.sum()
        z_den = np.sqrt(2 * (var ** 2).sum())
        z     = z_num / z_den
        p     = 2 * (1 - stats.norm.cdf(abs(z)))
        print(f"  z = {z:.3f},  p = {p:.4f}")
        if p > 0.05:
            print("  → Cannot reject H0: uncertainty estimates are well-calibrated.")
        else:
            print("  → Reject H0: uncertainty estimates are miscalibrated (p < 0.05).")

    #  Plots

    def plot_all(self):
        fig = plt.figure(figsize=(16, 12))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        #Predicted vs true scatter
        ax1 = fig.add_subplot(gs[0, 0])
        colors = {"severe": "#E24B4A", "moderate": "#BA7517", "mild": "#1D9E75", "normal": "#185FA5"}
        for bucket, color in colors.items():
            mask = self.df["bucket"] == bucket
            ax1.scatter(self.true[mask], self.pred[mask], c=color, alpha=0.5, s=15, label=bucket)
        lo_lim = min(self.true.min(), self.pred.min()) - 10
        hi_lim = max(self.true.max(), self.pred.max()) + 10
        ax1.plot([lo_lim, hi_lim], [lo_lim, hi_lim], "k--", lw=1, label="perfect")
        ax1.set_xlabel("True PF ratio (mmHg)")
        ax1.set_ylabel("Predicted PF ratio (mmHg)")
        ax1.set_title("Predicted vs True")
        ax1.legend(fontsize=8)

        #Residuals vs true
        ax2 = fig.add_subplot(gs[0, 1])
        residuals = self.pred - self.true
        ax2.scatter(self.true, residuals, alpha=0.4, s=15, color="#185FA5")
        ax2.axhline(0, color="k", lw=1, ls="--")
        ax2.set_xlabel("True PF ratio (mmHg)")
        ax2.set_ylabel("Residual (pred − true)")
        ax2.set_title("Residuals")

        #MAE per bucket bar chart
        ax3 = fig.add_subplot(gs[0, 2])
        bucket_names, bucket_maes = [], []
        for name, lo_pf, hi_pf in BUCKETS:
            mask = (self.true >= lo_pf) & (self.true < hi_pf)
            if mask.sum() >= 5:
                bucket_names.append(name)
                bucket_maes.append(self.mae(self.true[mask], self.pred[mask]))
        bar_colors = [colors[b] for b in bucket_names]
        ax3.bar(bucket_names, bucket_maes, color=bar_colors, edgecolor="white")
        ax3.set_ylabel("MAE (mmHg)")
        ax3.set_title("MAE per severity bucket")

        #Reliability diagram
        ax4 = fig.add_subplot(gs[1, 0])
        levels   = np.linspace(0.05, 0.99, 30)
        z_vals   = stats.norm.ppf((1 + levels) / 2)
        observed = []
        for z in z_vals:
            lo = np.exp(self.log_pred - z * self.log_pf_std)
            hi = np.exp(self.log_pred + z * self.log_pf_std)
            observed.append(np.mean((self.true >= lo) & (self.true <= hi)))
        ax4.plot(levels, observed, color="#185FA5", lw=2, label="model")
        ax4.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax4.set_xlabel("Nominal coverage")
        ax4.set_ylabel("Observed coverage")
        ax4.set_title("Reliability diagram")
        ax4.legend()

        #Prediction uncertainty (std) vs absolute error
        ax5 = fig.add_subplot(gs[1, 1])
        abs_err = np.abs(self.true - self.pred)
        ax5.scatter(self.std, abs_err, alpha=0.4, s=15, color="#534AB7")
        ax5.set_xlabel("Predicted std (σ)")
        ax5.set_ylabel("|True − Pred| (mmHg)")
        ax5.set_title("Uncertainty vs error\n(good model: positive correlation)")

        #Error distribution histogram
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.hist(residuals, bins=40, color="#185FA5", edgecolor="white", alpha=0.8)
        ax6.axvline(0, color="k", lw=1, ls="--")
        ax6.axvline(np.mean(residuals), color="#E24B4A", lw=1.5, ls="-", label=f"mean={np.mean(residuals):.1f}")
        ax6.set_xlabel("Residual (mmHg)")
        ax6.set_ylabel("Count")
        ax6.set_title("Residual distribution")
        ax6.legend(fontsize=8)

        plt.suptitle("PF Ratio Prediction — Evaluation", fontsize=14, fontweight="bold")
        path = f"{self.output_dir}/placeholder.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Plots saved to {path}")


    def run(self):
        print(f"\nEvaluating {len(self.df)} patients...")
        self.regression_metrics()
        self.bucket_metrics()
        self.coverage()
        self.spiegelhalter_test()
        self.plot_all()
        print("\nDone.")



def evaluate_from_csv(path: str, n_bootstrap: int = 1000, output_dir: str = "."):

    df = pd.read_csv(path)
    Evaluator(df, n_bootstrap=n_bootstrap, output_dir=output_dir).run()


#Classification Evaluator

def evaluate_classification_from_csv(path: str, output_dir: str = "."):
    """Evaluate classification predictions from Pipeline.predict() output."""
    from sklearn.metrics import (
        classification_report, accuracy_score,
        confusion_matrix, ConfusionMatrixDisplay
    )

    df = pd.read_csv(path).dropna(subset=["true_class", "pred_class"])
    y_true = df["true_class"].astype(int).values
    y_pred = df["pred_class"].astype(int).values
    names  = ["severe", "moderate", "mild", "normal"]

    acc = accuracy_score(y_true, y_pred)
    print(f"\nEvaluating {len(df)} patients...")
    print(f"\nOverall accuracy: {acc*100:.1f}%")
    print()
    print(classification_report(y_true, y_pred, target_names=names, zero_division=0))

    # confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — ARDS Severity Classification")
    plt.tight_layout()
    path_out = f"{output_dir}/placeholder.png"
    plt.savefig(path_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {path_out}")
    print("\nDone.")