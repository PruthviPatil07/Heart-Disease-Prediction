# Heart Disease Prediction

## Overview

This project is an end-to-end machine learning pipeline that predicts heart disease using various classification models. It includes data collection, preprocessing, model training, hyperparameter tuning, and deployment using Streamlit.

## Project Structure

```
BOOTCAMP
│── data
│   ├── raw_data
│   │   └── raw.csv
│   ├── cleaned_data
│   │   └── cleaned.csv
│   ├── preprocess
│       ├── x_train.csv
│       ├── x_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│── models
│   ├── best_model
│   ├── model
│── notebooks
│── src
│   ├── datacollection.py
│   ├── datacleaning.py
│   ├── preprocess.py
│   ├── modelbuilding.py
│   ├── hyperparameter.py
│── venv
│── app.py
│── Heart_Disease_Prediction.ipynb
│── streamlit.py
│── README.md
│── x.ipynb
```

## Machine Learning Pipeline

### Step 1: Data Collection

- The raw dataset is collected and stored in `data/raw_data/raw.csv`.
- The script `datacollection.py` is responsible for loading and validating the dataset.

### Step 2: Data Cleaning

- The dataset is cleaned using `datacleaning.py`.
- This step involves handling missing values, removing duplicates, and correcting inconsistencies.
- The cleaned data is stored in `data/cleaned_data/cleaned.csv`.

### Step 3: Data Preprocessing

- The dataset is split into training and testing sets using `preprocess.py`.
- Features and labels are separated and stored in `data/preprocess/` as:
  - `x_train.csv`, `x_test.csv` (features)
  - `y_train.csv`, `y_test.csv` (labels)
- Feature scaling and encoding are also handled in this step.

### Step 4: Model Training

- The script `modelbuilding.py` trains multiple models:
  - **Decision Tree Classifier**
  - **Random Forest Classifier**
  - **AdaBoost Classifier**
- These models are trained on `x_train.csv` and `y_train.csv`.
- The trained models are saved in the `models/model` directory.

### Step 5: Hyperparameter Tuning

- The script `hyperparameter.py` performs hyperparameter tuning to find the best model.
- The best model is selected based on performance metrics.
- The best model is saved in `models/best_model`.

### Step 6: Model Deployment

- The best model is loaded in `app.py` for making predictions.
- The `streamlit.py` script creates a web-based interface for user interaction.
- Users can input medical data to get heart disease predictions.

## How to Run the Project

### Prerequisites

Ensure you have Python installed along with the required dependencies:

```
pip install -r requirements.txt
```

### Steps to Run

1. Clone the repository:
   ```sh
   git clone https://github.com/PruthviPatil07/ML_Bootcamp.git
   cd ML_Bootcamp
   ```
2. Run the data pipeline:
   ```sh
   python src/datacollection.py
   python src/datacleaning.py
   python src/preprocess.py
   ```
3. Train the models:
   ```sh
   python src/modelbuilding.py
   ```
4. Perform hyperparameter tuning:
   ```sh
   python src/hyperparameter.py
   ```
5. Start the Streamlit app:
   ```sh
   streamlit run streamlit.py
   ```

## Deployment

The model is deployed using Streamlit, which provides a web-based interface for predictions.

## Acknowledgments

This project was developed as part of an ML bootcamp to demonstrate an end-to-end ML pipeline from data collection to deployment.

---

For more details, visit [GitHub Repository](https://github.com/PruthviPatil07/ML_Bootcamp).

