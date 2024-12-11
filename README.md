# Weather Prediction using Machine Learning

## Project Structure
```
weather-prediction/
├── data/              # Raw data and preprocessed data
│   ├── raw/           # Original, unprocessed data files (e.g., weather_data.csv)
│   └── processed/     # Preprocessed data files (e.g., weather_data_processed.csv)
├── src/               # Source code
│   ├── preprocessing.py
│   ├── models.py      # Could be split further (e.g., random_forest.py, xgboost.py, etc.)
│   ├── utils.py       # Helper functions (data handling, feature scaling, etc.)
│   ├── visualization.py # Visualization functions
│   ├── evaluation.py  # Model evaluation metrics and plotting
│   └── main.py
├── results/           # Output files (plots, model evaluation metrics, predictions)
│   ├── confusion_matrices/
│   ├── feature_importance/
│   ├── model_performance.csv # Summary table of model performance
│   └── predictions/       # Store model predictions (if needed)
├── reports/            # Project reports and documentation
│   └── report.pdf      # Main project report
└── README.md          # This file
```
## Getting Started

1. Clone the repository.
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the main script: `python src/main.py` (or adjust as necessary)


## Authors

* Waqas
* Atif
