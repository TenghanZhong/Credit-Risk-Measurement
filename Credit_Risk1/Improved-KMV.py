# Import required libraries
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from sklearn.neural_network import MLPRegressor


# Define a function to calculate asset value and asset volatility based on the option pricing formula
def asset_value_volatility(E, D, r, T, sigma_E):
    # E: Company equity value
    # D: Company's debt value at maturity
    # r: Risk-free interest rate
    # T: Debt maturity time
    # sigma_E: Stock volatility

    # Define a function to calculate option price and delta
    def black_scholes(S, K, r, T, sigma):
        # S: Underlying asset price
        # K: Strike price
        # r: Risk-free interest rate
        # T: Time to maturity
        # sigma: Asset volatility

        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Calculate the cumulative distribution function of the normal distribution
        def norm_cdf(x):
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        # Calculate the option price and delta
        c = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
        delta = norm_cdf(d1)

        return c, delta

    # Define a function to solve for asset value and asset volatility
    def solve_asset_value_volatility(x):
        # x: Variables to be solved, including asset value V and asset volatility sigma_V

        V = x[0]
        sigma_V = x[1]

        # Calculate the equity value and delta using the option pricing formula
        c, delta = black_scholes(V, D, r, T, sigma_V)

        # Define two equations and set them to zero
        f1 = c - E  # Option price minus equity value (since option price equals the company's equity value)
        f2 = delta * V - E  # Delta multiplied by asset value equals the company's equity value

        return [f1, f2]

    # Provide an initial guess and use fsolve to solve the system of equations
    x0 = [E + D,
          sigma_E]  # Initial guess: asset value equals equity value plus debt, asset volatility equals stock volatility
    x = fsolve(solve_asset_value_volatility, x0)  # Solve the system of equations

    return x[0], x[1]  # Return asset value and asset volatility


# Define a function to set the default point by weighting the remaining debt maturity
def default_point(ST_Debt, LT_Debt, LT_Maturity):
    # ST_Debt: Short-term debt value
    # LT_Debt: Long-term debt value
    # LT_Maturity: Remaining maturity of long-term debt

    # Calculate the weighting factor for long-term debt, assuming it matures evenly each year
    alpha = 1 / LT_Maturity

    # Compute the default point: short-term debt plus the weighted long-term debt
    D = ST_Debt + alpha * LT_Debt

    return D


# Define a function to build a mapping function using a neural network
def neural_network_mapping(DD):
    # DD: Distance to default

    # Assume there is a pre-trained neural network regressor named mlp
    # mlp = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', solver='adam', max_iter=1000)
    # mlp.fit(X_train, y_train)  # X_train is the training data for distance to default, y_train is the training data for default probability

    # Use the neural network regressor to predict default probability
    EDF = mlp.predict(DD)

    return EDF


# Define a function to describe the complex correlation structure between firms' asset values using a random correlation model
def random_correlation_model(V):
    # V: Firms' asset values

    # Assume there is a random correlation matrix named R
    # R = np.random.rand(n, n)  # n is the number of firms
    # R = (R + R.T) / 2  # Make R a symmetric matrix
    # R = R / np.sqrt(np.diag(R))  # Normalize R so that the diagonal elements are 1

    # Calculate the correlation between firms' asset values using the random correlation matrix
    rho = np.dot(V, np.dot(R, V.T))

    return rho


# Read the data file, assuming the file name is data.csv, which contains the following columns:
# Security Name, Stock Market Value, Debt Value at Maturity, Risk-Free Interest Rate, Debt Maturity, Stock Volatility, Remaining Maturity of Long-term Debt
df = pd.read_csv("C:\\Users\\zth020906\\Desktop\\工作簿2.csv", encoding='gbk')

E = data["收盘价"] * data["流通股"] + data["收盘价"] * data["非流通股"]  # Company market value, unit: ten thousand yuan

# For each row of data, call the function to calculate asset value and asset volatility, and add them as new columns
df["asset_value"] = df.apply(lambda row: asset_value_volatility(row["股票市值"], row["到期债务市值"], row["无风险利率"],
                                                                row["债务到期时间"], row["股票波动率"])[0], axis=1)
df["asset_volatility"] = df.apply(lambda row:
                                  asset_value_volatility(row["股票市值"], row["到期债务市值"], row["无风险利率"],
                                                         row["债务到期时间"], row["股票波动率"])[1], axis=1)

# For each row of data, call the function to set the default point based on the weighted remaining debt maturity, and add it as a new column
df["default_point"] = df.apply(
    lambda row: default_point(row["到期债务市值"], row["长期债务市值"], row["长期债务剩余期限"]), axis=1)

# For each row of data, calculate the distance to default and add it as a new column
df["default_distance"] = np.log(df["asset_value"] / df["default_point"]) + (
            df["无风险利率"] - 0.5 * df["asset_volatility"] ** 2) * df["债务到期时间"] / df["asset_volatility"]

# For each row of data, call the function to build the mapping using a neural network, and add it as a new column
df["default_probability"] = df.apply(lambda row: neural_network_mapping(row["default_distance"]), axis=1)

# Output the results
print(df)

# Import matplotlib library
import matplotlib.pyplot as plt

# Plot a scatter plot of distance to default vs. default probability
plt.scatter(df["default_distance"], df["default_probability"])
plt.xlabel("Default Distance")
plt.ylabel("Default Probability")
plt.title("Default Distance vs Default Probability")
plt.show()

# Plot a bar chart of security names vs. default probability
plt.bar(df["证券简称"], df["default_probability"])
plt.xlabel("Security Name")
plt.ylabel("Default Probability")
plt.title("Default Probability by Security Name")
plt.show()
