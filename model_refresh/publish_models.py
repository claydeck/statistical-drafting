"""Publish trained models to GitHub Releases.

Packages all ONNX models and card CSVs into a ZIP archive and uploads
it as a GitHub Release using the `gh` CLI.

Usage:
    python model_refresh/publish_models.py [--tag TAG]

Requires:
    - GitHub CLI (`gh`) installed and authenticated
    - Repository must have push access
"""

import argparse
import os
import zipfile
from datetime import date

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ONNX_DIR = os.path.join(DATA_DIR, "onnx")
CARDS_DIR = os.path.join(DATA_DIR, "cards")


def create_models_zip(output_path: str) -> int:
    """Create a ZIP archive containing all ONNX models and card CSVs.

    Returns the number of files added.
    """
    count = 0
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(ONNX_DIR)):
            if fname.endswith(".onnx"):
                zf.write(os.path.join(ONNX_DIR, fname), f"onnx/{fname}")
                count += 1
        for fname in sorted(os.listdir(CARDS_DIR)):
            if fname.endswith(".csv"):
                zf.write(os.path.join(CARDS_DIR, fname), f"cards/{fname}")
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Publish models to GitHub Releases")
    parser.add_argument("--tag", default=f"models-{date.today().isoformat()}",
                        help="Release tag (default: models-YYYY-MM-DD)")
    args = parser.parse_args()

    zip_path = os.path.join(DATA_DIR, "models.zip")
    print(f"Packaging models into {zip_path}...")
    count = create_models_zip(zip_path)
    print(f"Added {count} files to archive.")

    title = f"Models {args.tag}"
    print(f"Creating release {args.tag}...")
    exit_code = os.system(
        f'gh release create "{args.tag}" "{zip_path}" '
        f'--title "{title}" '
        f'--notes "Automated model release: {count} model files"'
    )

    if exit_code == 0:
        print(f"Release {args.tag} published successfully.")
    else:
        print(f"Failed to create release (exit code {exit_code}).")
        raise SystemExit(1)

    # Clean up
    os.remove(zip_path)


if __name__ == "__main__":
    main()
