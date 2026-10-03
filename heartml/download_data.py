
import shutil
from pathlib import Path

DATASETS = [
    "johnsmith88/heart-disease-dataset",
    "rashikrahmanpritom/heart-attack-analysis-prediction-dataset",
    "cherngs/heart-disease-cleveland",
    "fedesoriano/heart-failure-prediction",
]


def main(dest="data"):
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("pip install kagglehub   (or download the CSVs manually, see data/README.md)")
    dest = Path(dest)
    dest.mkdir(exist_ok=True)
    for slug in DATASETS:
        try:
            folder = Path(kagglehub.dataset_download(slug))
        except Exception as exc:  # network / auth
            print(f"! {slug}: {exc}")
            continue
        for csv in folder.rglob("*.csv"):
            target = dest / f"{slug.replace('/', '__')}__{csv.name}"
            shutil.copy(csv, target)
            print(f"  {slug} -> {target}")
    print("\nNow run:  python -m heartml.train")


if __name__ == "__main__":
    main()
