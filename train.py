"""Generate synthetic data, train the predictor, and report metrics."""
from motionscore.simulation import generate_dataset
from motionscore.modeling import train_model, MODEL_PATH

def main():
    data = generate_dataset(n_runs=1200, seed=42)
    results = train_model(data)
    print(f"Generated runs: {len(data)}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Held-out MAE:  {results['mae']:.4f}")
    print(f"Held-out RMSE: {results['rmse']:.4f}")

if __name__ == "__main__":
    main()
