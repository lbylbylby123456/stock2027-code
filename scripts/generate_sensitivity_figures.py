"""Generate sensitivity-analysis figures for ReCaT.

This script contains only anonymized plotting code and the numerical values
reported in the paper. It does not require training data.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


# Use TrueType fonts in PDF outputs to avoid Type-3 font warnings.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
mpl.rcParams["axes.linewidth"] = 1.2
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["xtick.major.width"] = 1.2
mpl.rcParams["ytick.major.width"] = 1.2


COLORS = {
    "csi_rankic": "#2C7BB6",
    "csi_sharpe": "#D7191C",
    "sp_rankic": "#FDBF6F",
    "sp_sharpe": "#1B9E77",
}


def plot_dual_sensitivity(
    output_path: str | Path,
    x_values: list[str],
    xlabel: str,
    csi300_rankic: list[float],
    csi300_sharpe: list[float],
    sp500_rankic: list[float],
    sp500_sharpe: list[float],
) -> None:
    """Plot RankIC and Sharpe sensitivity for two datasets."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax_left = plt.subplots(figsize=(6, 4.5), dpi=300)

    # Left axis: RankIC.
    l1 = ax_left.plot(
        x_values,
        csi300_rankic,
        color=COLORS["csi_rankic"],
        marker="o",
        linestyle="-",
        linewidth=2.5,
        markersize=8,
        label="CSI300 RankIC",
    )
    l2 = ax_left.plot(
        x_values,
        sp500_rankic,
        color=COLORS["sp_rankic"],
        marker="^",
        linestyle="-",
        linewidth=2.5,
        markersize=8,
        label="S&P500 RankIC",
    )
    ax_left.set_ylabel("RankIC", fontsize=14, fontweight="bold")
    ax_left.set_xlabel(xlabel, fontsize=12)
    ax_left.tick_params(axis="both", labelsize=12)
    ax_left.grid(True, linestyle="--", alpha=0.4)

    all_rankic = csi300_rankic + sp500_rankic
    ax_left.set_ylim(min(all_rankic) - 0.01, max(all_rankic) + 0.02)

    # Right axis: Sharpe ratio.
    ax_right = ax_left.twinx()
    l3 = ax_right.plot(
        x_values,
        csi300_sharpe,
        color=COLORS["csi_sharpe"],
        marker="s",
        linestyle="--",
        linewidth=2.5,
        markersize=8,
        label="CSI300 Sharpe",
    )
    l4 = ax_right.plot(
        x_values,
        sp500_sharpe,
        color=COLORS["sp_sharpe"],
        marker="D",
        linestyle="--",
        linewidth=2.5,
        markersize=8,
        label="S&P500 Sharpe",
    )
    ax_right.set_ylabel("Sharpe Ratio", fontsize=14, fontweight="bold")
    ax_right.tick_params(axis="y", labelsize=12)

    all_sharpe = csi300_sharpe + sp500_sharpe
    ax_right.set_ylim(min(all_sharpe) - 0.4, max(all_sharpe) + 0.6)

    lines = l1 + l2 + l3 + l4
    labels = [line.get_label() for line in lines]
    ax_left.legend(lines, labels, loc="best", frameon=False, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "figures"

    # Learning-rate sensitivity.
    x_lr = [r"$10^{-4}$", r"$5\times10^{-4}$", r"$10^{-3}$", r"$5\times10^{-3}$", r"$10^{-2}$"]
    csi300_rankic_lr = [0.041, 0.043, 0.043, 0.042, 0.041]
    csi300_sharpe_lr = [2.16, 2.15, 2.19, 2.09, 2.11]
    sp500_rankic_lr = [0.025, 0.023, 0.025, 0.025, 0.024]
    sp500_sharpe_lr = [1.99, 1.97, 2.01, 1.98, 2.00]

    # Transformer-depth sensitivity.
    x_layer = ["1", "2", "3", "4", "5"]
    csi300_rankic_layer = [0.036, 0.042, 0.043, 0.041, 0.042]
    csi300_sharpe_layer = [1.95, 2.12, 2.19, 2.08, 2.21]
    sp500_rankic_layer = [0.020, 0.025, 0.025, 0.026, 0.025]
    sp500_sharpe_layer = [1.88, 1.99, 2.01, 1.97, 1.98]

    # Regime-number sensitivity.
    x_regime = ["1", "2", "3", "4", "5"]
    csi300_rankic_regime = [0.039, 0.042, 0.043, 0.041, 0.043]
    csi300_sharpe_regime = [1.98, 2.15, 2.19, 2.09, 2.11]
    sp500_rankic_regime = [0.022, 0.024, 0.025, 0.024, 0.025]
    sp500_sharpe_regime = [1.82, 1.98, 2.01, 1.90, 1.95]

    plot_dual_sensitivity(
        output_dir / "sensitivity_lr.pdf",
        x_lr,
        "Learning Rate",
        csi300_rankic_lr,
        csi300_sharpe_lr,
        sp500_rankic_lr,
        sp500_sharpe_lr,
    )
    plot_dual_sensitivity(
        output_dir / "sensitivity_layers.pdf",
        x_layer,
        "Number of Transformer Layers",
        csi300_rankic_layer,
        csi300_sharpe_layer,
        sp500_rankic_layer,
        sp500_sharpe_layer,
    )
    plot_dual_sensitivity(
        output_dir / "sensitivity_regimes.pdf",
        x_regime,
        r"Number of Regimes $K$",
        csi300_rankic_regime,
        csi300_sharpe_regime,
        sp500_rankic_regime,
        sp500_sharpe_regime,
    )

    print("All sensitivity plots generated successfully.")


if __name__ == "__main__":
    main()
