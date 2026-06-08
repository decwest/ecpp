from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export generated figures and tables to the TeX project")
    parser.add_argument("--source-dir", type=Path, default=Path("results/access_ecpp_fixed_speed"))
    parser.add_argument("--tex-project-dir", type=Path, default=Path("../tex_docker_environment/projects/fumiya_ieee_access"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures_dest = args.tex_project_dir / "generated" / "figures"
    tables_dest = args.tex_project_dir / "generated" / "tables"
    figures_dest.mkdir(parents=True, exist_ok=True)
    tables_dest.mkdir(parents=True, exist_ok=True)
    for src in (args.source_dir / "plots").glob("fixed_speed_*.*"):
        if src.suffix.lower() in {".pdf", ".png"}:
            shutil.copy2(src, figures_dest / src.name)
    for src in (args.source_dir / "tables").glob("fixed_speed_*.tex"):
        shutil.copy2(src, tables_dest / src.name)
    print(f"exported to: {args.tex_project_dir}")


if __name__ == "__main__":
    main()
