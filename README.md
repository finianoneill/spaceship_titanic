# Spaceship Titanic

This repo contains code used in the Kaggle competition "Spaceship Titanic". Most of the code will be contained within Jupyter Notebooks.

## Overview

TBD

## Getting Started

### Prerequisites

- Python 3.10+
- Jupyter Notebook
- Required packages:
  - pandas
  - scikit-learn
  - joblib
  - numpy

### Installation

```bash
# Clone the repository
git clone https://github.com/finianoneill/spaceship_titanic.git
cd spaceship_titanic

# Create and activate a conda environment
conda create -n spaceship-titanic python=3.12
conda activate spaceship-titanic

# Install dependencies
pip install -r requirements.txt

# macOS only: XGBoost/LightGBM need the OpenMP runtime
conda install -c conda-forge llvm-openmp
```

### Kaggle CLI Commands

To download the competition data using the Kaggle CLI:

```bash
# Make sure you have your Kaggle API token saved in ~/.kaggle/access_token (or legacy ~/.kaggle/kaggle.json)
kaggle competitions download -c spaceship-titanic
unzip spaceship-titanic.zip -d data/
```

## Project Structure

```
TBD
```

## Data Processing

TBD

## Model Selection

TBD

## Making Predictions

TBD

## Model Features

TBD

## Author

- Finian O'Neill

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- This project is part of a Kaggle competition
- Thanks to Kaggle for providing the dataset and platform