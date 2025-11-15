# 📈 Stock Price Prediction using Time Series Models

## Overview
This project demonstrates the use of time series modeling, typically implemented with **Recurrent Neural Networks (RNNs)** like **Long Short-Term Memory (LSTM)**, to predict future stock prices. The model learns patterns from historical price data to forecast movements in the stock market.

---

## Dataset
The project utilizes historical stock price data, with the initial analysis focusing on the training data:

* **Training Data File:** `Google_Stock_Price_Train.csv`.
* **Data Structure:** The dataset contains time-series features typical for stock data, such as `Date`, `Open`, `High`, `Low`, `Close`, and `Volume`.

A separate test dataset (e.g., `Google_Stock_Price_Test.csv`, typically used for final evaluation in this type of project) would be required to assess the model's performance on genuinely unseen data.

---

## Prerequisites (Dependencies)
The following libraries are necessary to run the notebook and execute the prediction model:

* **`pandas`** and **`numpy`**: For data loading and manipulation.
* **`matplotlib`** and **`seaborn`**: For data visualization and plotting the results.
* **`tensorflow`** / **`keras`**: (Inferred) Essential for building and training the deep learning LSTM model.
* **`scikit-learn`**: (Inferred) Used for data preprocessing tasks like scaling (e.g., `MinMaxScaler`) and preparing the data sequences.

---

## Methodology and Steps

The typical workflow for stock price prediction using a deep learning model involves specialized preprocessing for time series data:

### 1. Data Loading and Initial Analysis
* **Load Data:** The training data is loaded using `pandas.read_csv()`.
* **Exploration:** Initial checks like `df_train.head()` are performed to examine the first few rows and understand the feature set.

### 2. Time Series Preprocessing
* **Feature Selection:** The relevant price series (e.g., the 'Open' price) is extracted and used as the primary feature for prediction.
* **Feature Scaling:** The data is scaled (e.g., using **MinMaxScaler**) to normalize values between 0 and 1. This is critical for stabilizing and improving the convergence of the LSTM model.
* **Sequence Creation:** The data is transformed into **time step sequences** (e.g., using the last 60 days to predict the price on the 61st day), converting the flat time series data into the 3D structure required for LSTM models (samples, time steps, features).

### 3. Model Building and Training
* **Model Architecture:** A Sequential model containing one or more **LSTM layers** is constructed. Dropout layers are often included to prevent overfitting.
* **Output Layer:** The model concludes with a Dense output layer, typically with a single unit for predicting a single continuous price value.
* **Compilation:** The model is compiled using an appropriate optimizer (e.g., **Adam**) and a loss function suitable for regression (e.g., **Mean Squared Error (MSE)**).
* **Training:** The model is trained on the prepared time step sequences for a set number of epochs.

### 4. Evaluation and Prediction
* **Test Data Preparation:** The test dataset is loaded, preprocessed, and scaled *using the scaler fitted on the training data*.
* **Prediction:** The trained LSTM model generates predictions on the test set.
* **Inverse Transformation:** The scaled predictions are inverse-transformed back to the original dollar values.
* **Visualization:** The final predicted stock prices are plotted against the actual test prices to visually assess the model's accuracy and fit.
