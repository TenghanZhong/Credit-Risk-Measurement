# Import numpy library
import numpy as np

# Define the credit rating transition probability matrix
convert_rate_matrix = np.array([
    [0.9993, 0.0005, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0008, 0.9915, 0.0067, 0.0006, 0.0002, 0.0001, 0.0001],
    [0.0000, 0.0030, 0.8667, 0.0533, 0.0127, 0.0027, 0.0027],
    [0.0000, 0.0003, 0.0313, 0.7500, 0.1875, 0.0094, 0.0031],
    [1.e-2, 1.e-2, 7.e-2, 7.e-2, 8.e-2, 71.e-2, 11.e-2],
    [9.e-3, 9.e-3, 9.e-3, 9.e-3, 8.e-2, 83.e-2, 4.e-2],
    [17.e-4, 17.e-4, 17.e-4, 17.e-4, 17.e-4, 82.e-3, 82.e-3]
])

# Define the forward discount rates corresponding to credit ratings
discount_rate = np.array([
    [4.41e-2, 4.72e-2, 5.3e-2],
    [4.81e-2, 5.12e-2, 5.7e-2],
    [5.51e-2, 5.92e-2, 6.3e-2],
    [7.21e-2, 7.52e-2, 8.1e-2],
    [11.e-1, 13.e-1, 15.e-1]
])

# Define the recovery rates corresponding to credit ratings
recovery_rate = np.array([51.e-2 for i in range(7)])

# Define the correlation matrix for bonds
correlation_matrix = np.array([
    [1, 0.89, 0.84],
    [0.89, 1, 0.79],
    [0.84, 0.79, 1]
])

# Define bond face value, coupon rate, and issue size
face_value = np.array([100 for i in range(3)])
coupon_rate = np.array([3.e-8, 4.e-1, 4.e-3])
issue_size = np.array([50.e8, 30.e8, 20.e8])

# Define the portfolio weights
weight = np.array([33333 / 100 for i in range(3)])

# Define the credit rating labels
credit_label = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-",
                "CCC"]


# Define a function to calculate bond value
def bond_value(face_value, coupon_rate, duration, r):
    # face_value: Bond face value
    # coupon_rate: Bond coupon rate
    # duration: Bond remaining term
    # r: Required return rate for the bond
    # return: Bond value

    value = face_value * coupon_rate  # Initialize bond value as the first year's interest income
    for i in range(1, duration):  # Loop through remaining years to calculate present value
        value += face_value * coupon_rate / (1 + r[i]) ** i  # Add the present value of the annual interest
    value += face_value / (
                1 + r[-1]) ** duration  # Add the present value of the principal and interest in the final year
    return value


# Define a function to calculate portfolio value
def portfolio_value(face_value, coupon_rate, duration, r, recovery_rate):
    # face_value: Array of bond face values
    # coupon_rate: Array of bond coupon rates
    # duration: Array of bond remaining terms
    # r: Array of required return rates for bonds
    # recovery_rate: Array of bond recovery rates
    # return: Portfolio value

    value = np.zeros((len(face_value),
                      len(r)))  # Initialize the portfolio value matrix as a zero matrix; rows: number of bonds, columns: number of credit ratings
    for i in range(len(face_value)):  # Loop through each bond to calculate its value for each credit rating
        for j in range(len(r)):
            if j == len(
                    r) - 1:  # If it is the last column (i.e., default case), then the value is face value multiplied by the recovery rate
                value[i, j] = face_value[i] * recovery_rate[j]
            else:  # Otherwise, use the bond_value function to calculate the bond value
                value[i, j] = bond_value(face_value[i], coupon_rate[i], duration[i], r[j])

    value = value * weight.reshape(-1,
                                   1)  # Multiply the value of each bond by its corresponding weight to obtain the portfolio's value distribution across credit ratings
    value = value.sum(axis=0)  # Sum each column to get the total portfolio value for each credit rating

    return value


# Define a function to calculate portfolio Value at Risk (VAR)
def portfolio_var(value, p):
    # value: Array of portfolio values across credit ratings
    # p: Confidence level (percentile)
    # return: Portfolio Value at Risk (VAR)

    var = np.percentile(value, p) - np.mean(value)  # Calculate VAR as the p-th percentile minus the mean

    return var


value = portfolio_value(face_value, coupon_rate, [3 for i in range(3)], discount_rate, recovery_rate)
print("The portfolio value for each credit rating is:")
for i in range(len(value)):
    print(credit_label[i], ":", value[i])
# Call the portfolio_var function to calculate the portfolio Value at Risk (VAR) and print the result
var = portfolio_var(value, [95])
print("The portfolio Value at Risk (VAR) at 95% confidence level is:")
print(var)
