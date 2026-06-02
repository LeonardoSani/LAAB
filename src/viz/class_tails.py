import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_class_tails(df, save_path: Path) -> None:
    """Figure 6 (E2): digit-class composition of good/bad tails.

    Per class, two bars: good (green) and bad (red). Within each, the light
    bar is the 20% tail count, the strong bar the 10% subset overlaid on top
    (10% is nested in 20%). A faint marker shows the class size (ceiling)."""
    classes = df["class"].to_numpy()
    x = np.arange(len(classes))
    w = 0.4

    fig, ax = plt.subplots(figsize=(11, 5))

    # good (green): light=20%, strong=10% overlaid
    ax.bar(x - w / 2, df["good20"], w, color="green", alpha=0.30,
           label="best 20%")
    ax.bar(x - w / 2, df["good10"], w, color="green", alpha=1.0,
           label="best 10%")
    # bad (red): light=20%, strong=10% overlaid
    ax.bar(x + w / 2, df["bad20"], w, color="red", alpha=0.30,
           label="worst 20%")
    ax.bar(x + w / 2, df["bad10"], w, color="red", alpha=1.0,
           label="worst 10%")

    # class-size ceiling reference
    ax.plot(x, df["n_class"], "k_", ms=18, alpha=0.5, label="anchors in class")

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_xlabel("digit class")
    ax.set_ylabel("# anchors of class in tail")
    ax.set_title(r"E2 — Class composition of good/bad anchor tails ($t=\infty$)")
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
