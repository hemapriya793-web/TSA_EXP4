# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 31.10.2026



### AIM:
To implement ARMA model in python.
dataset:super store sale
software required:google colab
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 1. Dataset Loading and Fallback Setup
file_path = "superstore_final_dataset(1).csv"
if os.path.exists(file_path):
    print(f"Loading original dataset from '{file_path}'...")
    data = pd.read_csv(file_path, encoding="cp1252")
    X = data['Sales'].dropna()
    target_name = 'Sales'
else:
    print(f"'{file_path}' not found in working directory. Using Colab sample data as fallback.")
    data = pd.read_csv("sample_data/california_housing_train.csv")
    X = data['median_house_value'].dropna()
    target_name = 'Median House Value'

# Configure global plotting parameters
N = 1000
plt.rcParams['figure.figsize'] = [12, 6]

# 2. Plot Original Target Data
plt.figure()
plt.plot(X.values)
plt.title(f'Original {target_name} Data')
plt.xlabel('Index')
plt.ylabel(target_name)
plt.show()

# 3. Plot ACF and PACF for Original Data
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(X, lags=35, ax=axes[0])
axes[0].set_title(f'{target_name} Autocorrelation Function (ACF)')

plot_pacf(X, lags=35, ax=axes[1])
axes[1].set_title(f'{target_name} Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

# 4. ARMA(1,1) Model Fitting & Simulation
print("Fitting ARMA(1,1) model...")
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

phi1_arma11 = arma11_model.arparams[0]
theta1_arma11 = arma11_model.maparams[0]

# Generate simulated ARMA(1,1) process
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

# Plot ARMA(1,1) results
plt.figure()
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(ARMA_1, lags=35, ax=axes[0])
axes[0].set_title('ACF - Simulated ARMA(1,1)')
plot_pacf(ARMA_1, lags=35, ax=axes[1])
axes[1].set_title('PACF - Simulated ARMA(1,1)')
plt.tight_layout()
plt.show()

# 5. ARMA(2,2) Model Fitting & Simulation
print("Fitting ARMA(2,2) model...")
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

phi1_arma22 = arma22_model.arparams[0]
phi2_arma22 = arma22_model.arparams[1]
theta1_arma22 = arma22_model.maparams[0]
theta2_arma22 = arma22_model.maparams[1]

# Generate simulated ARMA(2,2) process
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 10)

# Plot ARMA(2,2) results
plt.figure()
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(ARMA_2, lags=35, ax=axes[0])
axes[0].set_title('ACF - Simulated ARMA(2,2)')
plot_pacf(ARMA_2, lags=35, ax=axes[1])
axes[1].set_title('PACF - Simulated ARMA(2,2)')
plt.tight_layout()
plt.show()
```
OUTPUT:
<img width="1036" height="547" alt="image" src="https://github.com/user-attachments/assets/13373c78-5c90-4dd1-a9e7-5634607665a8" />

<img width="993" height="528" alt="image" src="https://github.com/user-attachments/assets/dc611c66-009a-4522-aa25-0dfcce1cd01a" />

<img width="993" height="528" alt="image" src="https://github.com/user-attachments/assets/25252f67-1cf9-4baa-938b-4673213a3a60" />



RESULT:
Thus, a python program is created to fir ARMA Model successfully.
