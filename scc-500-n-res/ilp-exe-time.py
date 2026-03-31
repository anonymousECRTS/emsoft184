import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def parse_times_file(path: str) -> list[float]:
    values = []
    pattern = re.compile(r"\[([0-9]*\.?[0-9]+)\]")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                values.append(float(m.group(1)))
    if not values:
        raise ValueError(f"No values found in {path}")
    return values


def extract_info(path: str) -> tuple[str, int]:
    """
    Returns:
        variant: 'base' or '3'
        scc_count: integer from filename

    Accepted filenames:
        scc_times-n-1.txt
        scc_times-n-3-1.txt
    """
    name = Path(path).name
    m = re.search(r"scc_times-n-(3-)?(\d+)\.txt$", name)
    if not m:
        raise ValueError(f"Cannot extract info from filename: {name}")

    variant = "3" if m.group(1) else "base"
    scc_count = int(m.group(2))
    return variant, scc_count


def bootstrap_mean_ci(
    data: list[float],
    n_bootstrap: int = 10000,
    ci: float = 95.0,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(data, dtype=float)
    mean = float(np.mean(arr))

    boot_means = np.empty(n_bootstrap, dtype=float)
    n = len(arr)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = (100.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha))
    upper = float(np.percentile(boot_means, 100.0 - alpha))
    return mean, lower, upper


def collect_stats(files: list[str]) -> dict[str, dict[str, np.ndarray]]:
    grouped = {"base": [], "3": []}

    for path in files:
        variant, scc = extract_info(path)
        values = parse_times_file(path)
        mean, lower, upper = bootstrap_mean_ci(values)

        grouped[variant].append((scc, mean, lower, upper, len(values)))

        print(
            f"{variant:>4} | SCC={scc:>3} | n={len(values):>3} | "
            f"mean={mean:.4f} | 95% CI=({lower:.4f}, {upper:.4f})"
        )

    result = {}
    for variant, rows in grouped.items():
        if not rows:
            continue

        rows.sort(key=lambda x: x[0])

        result[variant] = {
            "scc_counts": np.array([r[0] for r in rows]),
            "means": np.array([r[1] for r in rows]),
            "lowers": np.array([r[2] for r in rows]),
            "uppers": np.array([r[3] for r in rows]),
        }

    return result


def main():
    files = glob.glob("scc_times-n-[0-9]*.txt") + glob.glob("scc_times-n-3-*.txt")
    if not files:
        raise FileNotFoundError(
            "No files matching 'scc_times-n-*.txt' or 'scc_times-n-3-*.txt' found."
        )

    stats = collect_stats(files)

    plt.figure(figsize=(8, 5))

    if "base" in stats:
        x = stats["base"]["scc_counts"]
        y = stats["base"]["means"]
        lo = stats["base"]["lowers"]
        hi = stats["base"]["uppers"]

        plt.plot(x, y, marker="o", label="Networks with initial token factor 2")
        plt.fill_between(x, lo, hi, alpha=0.2)

    if "3" in stats:
        x = stats["3"]["scc_counts"]
        y = stats["3"]["means"]
        lo = stats["3"]["lowers"]
        hi = stats["3"]["uppers"]

        plt.plot(x, y, marker="s", label="Networks with initial token factor 3")
        plt.fill_between(x, lo, hi, alpha=0.2)

    all_x = sorted(
        set(stats.get("base", {}).get("scc_counts", []))
        | set(stats.get("3", {}).get("scc_counts", []))
    )

    plt.xlabel("Number of SCCs", fontsize=18)
    plt.ylabel("Runtime (s)", fontsize=18)
    plt.xticks(all_x, fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("scc_runtime_ci_combined.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()