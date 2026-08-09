import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fungsi_linier(X: np.ndarray, gradien: np.float64) -> np.ndarray:
    return X * gradien


try:
    # Generate Data
    data = pd.read_csv("bensin.csv")
    X = data[["Liter"]].to_numpy().squeeze()
    y = data[["Kilometer"]].to_numpy().squeeze()

    # Nilai Slope awal
    m_init = np.float64(10.0)

    # Mencari Nilai m 
    m_prediction = m_init

    # Nilai intercept atau bias awal
    intercept_or_bias = 0

    # Data Prediksi yang belum di training
    X_prediction  = np.linspace(1, 45, len(X))
    y_prediction = fungsi_linier(X_prediction, m_prediction) + intercept_or_bias

    # Nilai error awal
    error = 0

    print(f"Nilai Slope, Intercept dan Error yang BELUM di Training")
    print(f"================================================")
    print(f"Slope\t\t: {m_prediction}")
    print(f"Intercept\t: {intercept_or_bias}")
    print(f"Error\t\t: {error}")
    print("")

    # Data Prediksi yang sedang di training
    for i in range(0,37):
        # global y_prediction_new
        y_prediction_new = fungsi_linier(np.array(X[i], dtype=np.ndarray), np.float64(m_prediction))
        y_actual = np.array(y[i], dtype=np.ndarray)


        error = y_actual - y_prediction_new
        delta_m = error / np.array(X[i], dtype=np.ndarray)
        m_prediction = m_prediction + delta_m

        intercept_or_bias = y_prediction_new - fungsi_linier(np.array(X[i], dtype=np.ndarray), np.float64(m_prediction))

    # Data Prediksi yang sudah di training
    X_prediction = np.linspace(1, 45, len(X))
    y_prediction = fungsi_linier(X_prediction, np.float64(m_prediction)) + intercept_or_bias


    print(f"Nilai Slope, Intercept dan Error yang SUDAH di Training")
    print(f"================================================")
    # Nilai Slope atau m
    print(f"Slope\t\t: {m_prediction}")
    # Nilai Intercept atau Bias
    print(f"Intercept\t: {intercept_or_bias}")
    # Nilai error yang sudah di training
    print(f"Error\t\t: {error}")

    # Visualisasi Data
    plt.scatter(X, y)
    plt.scatter(X_prediction, y_prediction, c="red")
    plt.xlabel("Liter")
    plt.ylabel("Jarak (km)")
    plt.legend(["Data Asli", "Data Prediksi"])
    plt.savefig("Result_Prediction.jpg")
    plt.show()
except FileNotFoundError:
    print("File tidak ditemukan")

