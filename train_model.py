import argparse
from sandbox.predictive_model import train_and_save

def main():
    parser = argparse.ArgumentParser(description="Train Emergent Behavior Sandbox Predictive Model")
    parser.add_argument("--db", type=str, default="simulation_runs.duckdb", help="Path to DuckDB database")
    args = parser.parse_args()

    print("=" * 60)
    print("        TRAINING PREDICTIVE FAILURE RISK MODEL")
    print("=" * 60)
    print(f"Database source: {args.db}")
    print("Fitting Random Forest classifier on historical features...")
    
    result = train_and_save(db_path=args.db)
    
    print("-" * 60)
    print("Training Complete!")
    print(f"Saved model path: {result['model_path']}")
    print(f"Total samples trained: {result['samples_trained']}")
    print(f"Model test accuracy: {result['accuracy']:.2%}")
    
    # Print classification report highlights
    report = result["report"]
    print("Detailed metrics:")
    if "0" in report:
        print(f"  - Clean runs (Class 0) F1-score: {report['0']['f1-score']:.2%}")
    if "1" in report:
        print(f"  - Failure runs (Class 1) F1-score: {report['1']['f1-score']:.2%}")
    print("=" * 60)

if __name__ == "__main__":
    main()
