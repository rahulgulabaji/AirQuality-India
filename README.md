# Air Quality Health Risk Prediction

This project is a machine learning application that predicts the health risk based on air quality data. It includes a Jupyter Notebook for data analysis and model training, and a Streamlit web application for interactive predictions.

## Features

- **Exploratory Data Analysis (EDA):** The project includes a detailed EDA of the air quality dataset, with visualizations to understand the data distribution, trends, and correlations.
- **Model Training:** Three different classification models (Logistic Regression, Decision Tree, and Random Forest) are trained and evaluated to predict the health risk.
- **Hyperparameter Tuning:** The Random Forest model is tuned using GridSearchCV to find the best hyperparameters and improve its performance.
- **Feature Importance:** The project visualizes the most important features for the prediction, providing insights into the factors that influence health risk.
- **Web Application:** A Streamlit web application is provided to interact with the trained model and get real-time predictions.
- **Testing:** The project includes unit tests for the web application to ensure its correctness.

## File Structure

- `health_risk_prediction.ipynb`: The main Jupyter Notebook containing the data analysis, model training, and evaluation.
- `app.py`: The Streamlit web application.
- `test_app.py`: Unit tests for the web application.
- `city_day_filled.csv`: The dataset used for training the model.
- `requirements.txt`: A list of all the required Python libraries.
- `health_risk_model.pkl`: The saved trained machine learning model.
- `scaler.pkl`: The saved scaler for preprocessing the input data.
- `le.pkl`: The saved label encoder for the target variable.
- `columns.json`: A list of the columns used for training the model.

## Getting Started

### Prerequisites

- Python 3.x
- pip

### Installation

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required libraries using the `requirements.txt` file:

   ```
   pip install -r requirements.txt
   ```

## How to Run

### Jupyter Notebook

To run the Jupyter Notebook, you will need to have Jupyter Notebook or JupyterLab installed. You can then run the following command in your terminal:

```
jupyter notebook health_risk_prediction.ipynb
```

Then, run all the cells in the notebook from top to bottom. This will train the model and save the necessary files for the web application.

### Streamlit App

After running the Jupyter Notebook and generating the model files, you can run the Streamlit web application with the following command:

```
streamlit run app.py
```

Or, if the `streamlit` command is not found, you can use:

```
python -m streamlit run app.py
```

## How to Use the App

Once the Streamlit app is running, you can:

1.  Enter the values for the air quality features in the input fields.
2.  Select a city from the dropdown menu.
3.  Click the "Predict" button to see the predicted health risk.

## Testing

To run the unit tests for the application, you can use `pytest`:

```
pytest test_app.py
```

## Screenshots

### Main Page

![Main Page](screenshots/Screenshot%202025-09-13%20123024.png)

### Prediction Result

![Prediction Result](screenshots/Screenshot%202025-09-13%20123515.png)

![Prediction Result](screenshots/Screenshot%202025-09-13%20123529.png)

![Prediction Result](screenshots/Screenshot%202025-09-13%20123553.png)