# Bitcoin Transaction Anomaly Detection

## Overview

This project implements a machine learning system for detecting anomalous (potentially illicit) transactions in the Bitcoin network. It utilizes the Elliptic Data Set, which contains features from Bitcoin transactions, some of which are labeled as licit or illicit.

## Table of Contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Data Description](#data-description)
7. [Model](#model)
8. [Visualization](#visualization)
9. [Results](#results)
10. [Contributing](#contributing)
11. [License](#license)

## Features

- Data preprocessing and feature engineering
- Machine learning model for anomaly detection
- Visualization of transaction data and model results
- Performance evaluation metrics

## Requirements

- Python 3.7+
- pip (Python package installer)

## Installation

1. Clone or download the repository to your local machine.

2. Navigate to the project directory:
   ```
   cd path/to/bitcoin-anomaly-detection
   ```

3. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Ensure your data files (elliptic_txs_features.csv, elliptic_txs_classes.csv, elliptic_txs_edgelist.csv) are in the `data` directory.

2. Run the main script:
   ```
   python main.py
   ```

3. View the results in the console output and the generated visualization files.

## Project Structure

- `main.py`: The entry point of the application
- `config.py`: Configuration settings for the project
- `data_loader.py`: Functions for loading and preprocessing data
- `feature_engineering.py`: Feature engineering and selection
- `model_training.py`: Machine learning model definition and training
- `anomaly_detector.py`: Functions for detecting anomalies in transactions
- `visualization.py`: Functions for creating visualizations
- `utils.py`: Utility functions used across the project
- `requirements.txt`: List of Python package dependencies

## Data Description

The Elliptic Data Set consists of:
- 203,769 node transactions
- 234,355 edges (flows between transactions)
- 166 features per transaction
- 49 time steps

Class distribution:
- Illicit: 2% (4,545 transactions)
- Licit: 21% (42,019 transactions)
- Unknown: 77% (157,205 transactions)

## Model

The project uses a Random Forest classifier for anomaly detection. The model is trained on labeled data and evaluated using metrics such as precision, recall, and F1-score.

## Visualization

The project generates several visualizations:
- t-SNE plot of transactions
- Feature importance bar chart
- Anomaly score distribution histogram

## Results

Results are output to the console and saved in `final_results.csv`. This includes classification reports and performance metrics for the training, validation, and test sets.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is distributed under the [MIT License](LICENSE).

---

For more detailed information about the project architecture and specifications, please refer to the `project_specification.md` file in the project root directory.

