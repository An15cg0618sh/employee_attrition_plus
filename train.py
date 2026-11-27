from pathlib import Path
import json

from src.attrition import load_dataset, train_and_compare, save_model

DATA_PATH = Path("data/employee_data.csv")
ARTIFACT_DIR = Path("artifacts")

def main():
    data = load_dataset(DATA_PATH)
    results, comparison, cleaning_report = train_and_compare(data)
    ARTIFACT_DIR.mkdir(exist_ok=True)

    print("\nModel comparison")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best_name = comparison.iloc[0]["model"]
    save_model(results[best_name], ARTIFACT_DIR / "best_model.joblib")

    report = {
        "selected_model": best_name,
        "metrics": results[best_name].metrics,
        "cross_validation": results[best_name].cv_metrics,
        "data_quality": cleaning_report,
    }
    (ARTIFACT_DIR / "training_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved model and report under: {ARTIFACT_DIR.resolve()}")

if __name__ == "__main__":
    main()
