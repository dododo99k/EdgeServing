from pathlib import Path
import re


def rename_ours_normalized_pickles(
    logs_dir: str = "logs",
    new_name_template: str = "multi_model_diag_ours_normalized_lam{number}_old.pkl",
    dry_run: bool = False,
) -> None:
    """
    Rename multi_model_diag_ours_normalized_lam{number}.pkl in each lam152_{number}
    folder under logs_dir.

    new_name_template can include "{number}" to keep the lam number in the name.
    Example: "ours_normalized.pkl" or "ours_normalized_lam{number}.pkl"
    """
    root = Path(logs_dir)
    if not root.exists():
        raise FileNotFoundError(f"logs_dir not found: {root}")

    pattern = re.compile(r"^lam152_(\d+)$")
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue

        match = pattern.match(subdir.name)
        if not match:
            continue

        number = match.group(1)
        src = subdir / f"multi_model_diag_ours_normalized_lam{number}.pkl"
        if not src.exists():
            print(f"skip (missing): {src}")
            continue

        dst = subdir / new_name_template.format(number=number)
        if dst.exists():
            print(f"skip (exists): {dst}")
            continue

        if dry_run:
            print(f"would rename: {src} -> {dst}")
        else:
            src.rename(dst)
            print(f"renamed: {src} -> {dst}")


if __name__ == "__main__":
    rename_ours_normalized_pickles()
