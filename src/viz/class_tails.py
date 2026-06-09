import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_class_tails(df, save_path: Path) -> None:
    """Fig 6: digit-class composition of good/bad mu tails. Green/red bars,
    light = 20% tail, strong = 10% nested; marker = class size."""
    classes = df["class"].to_numpy()
    x = np.arange(len(classes))
    w = 0.4

    fig, ax = plt.subplots(figsize=(11, 5))

    # good (green) / bad (red): 20% light, 10% strong overlaid
    ax.bar(x - w / 2, df["good20"], w, color="green", alpha=0.30,
           label="best 20%")
    ax.bar(x - w / 2, df["good10"], w, color="green", alpha=1.0,
           label="best 10%")
    ax.bar(x + w / 2, df["bad20"], w, color="red", alpha=0.30,
           label="worst 20%")
    ax.bar(x + w / 2, df["bad10"], w, color="red", alpha=1.0,
           label="worst 10%")

    ax.plot(x, df["n_class"], "k_", ms=18, alpha=0.5, label="anchors in class")

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_xlabel("digit class")
    ax.set_ylabel("# anchors of class in tail")
    ax.set_title(r"Class composition of good/bad anchor tails ($t=\infty$)")
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
