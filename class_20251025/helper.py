"""
Helper functions for MLP_Class.ipynb
Contains utility functions for data generation, testing, and visualization
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_data(n=100):
    """
    Generate a simple 1-dimensional toy dataset for regression

    INPUT:
        n - number of samples to generate (default: 100)

    OUTPUT:
        X - feature matrix of shape (n, 2) where the second column is all ones (bias term)
        y - label vector of shape (n,)
    """
    # Generate x values between -3 and 3
    x = np.linspace(-3, 3, n)

    # Generate y values with a non-linear pattern plus some noise
    # Using a combination of polynomial and trigonometric functions
    y = 0.5 * x**2 - 0.3 * x**3 + 2 * np.sin(x) + np.random.randn(n) * 0.5

    # Append 1 to each feature vector for bias
    X = np.column_stack([x, np.ones(n)])

    return X, y


def runtest(test_function, test_name):
    """
    Test runner utility that executes test functions and displays results

    INPUT:
        test_function - a function that returns True if test passes, False otherwise
        test_name - string name of the test for display purposes
    """
    try:
        result = test_function()
        if result:
            print(f"✓ {test_name} passed")
        else:
            print(f"✗ {test_name} failed")
    except Exception as e:
        print(f"✗ {test_name} failed with exception: {str(e)}")


def ReLU(z):
    """ReLU activation function"""
    return np.maximum(z, 0)


def ReLU_grad(z):
    """Gradient of ReLU activation function"""
    return (z > 0).astype("float64")


def forward_pass_grader(W, xTr):
    """
    Reference implementation of forward pass for grading

    INPUT:
        W - an array of L weight matrices
        xTr - nxd matrix. Each row is an input vector

    OUTPUTS:
        A - a list of matrices (of length L) that stores result of matrix multiplication at each layer
        Z - a list of matrices (of length L) that stores result of transition function at each layer
    """
    # Initialize A and Z
    A = [xTr]
    Z = [xTr]

    for i, w in enumerate(W):
        A.append(Z[-1] @ w)
        if i < len(W) - 1:  # Apply ReLU to all layers except the last
            Z.append(ReLU(A[-1]))
        else:  # Last layer - no activation
            Z.append(A[-1])

    return A, Z


def MSE_grader(out, y):
    """
    Reference implementation of MSE loss for grading

    INPUT:
        out - output of network (n vector)
        y - training labels (n vector)

    OUTPUT:
        loss - the MSE loss (a scalar)
    """
    n = len(y)
    loss = np.sum(np.square(out - y)) / n
    return loss


def MSE_grad_grader(out, y):
    """
    Reference implementation of MSE gradient for grading

    INPUT:
        out - output of network (n vector)
        y - training labels (n vector)

    OUTPUT:
        grad - the gradient of the MSE loss with respect to out (n vector)
    """
    n = len(y)
    grad = 2 * (out - y) / n
    return grad


def backprop_grader(W, A, Z, y):
    """
    Reference implementation of backpropagation for grading

    INPUT:
        W - weights (cell array)
        A - output of forward pass (cell array)
        Z - output of forward pass (cell array)
        y - vector of size n (each entry is a label)

    OUTPUT:
        gradients - the gradient with respect to W as a list of matrices
    """
    # Convert delta to a column vector
    delta = (MSE_grad_grader(Z[-1].flatten(), y) * 1).reshape(-1, 1)

    # Compute gradient with backprop
    gradients = []
    for i in range(len(W) - 1, -1, -1):
        # Gradient for W[i]
        gradients.append(Z[i].T @ delta)

        # Update delta for next iteration (going backwards)
        if i > 0:
            delta = ReLU_grad(A[i]) * (delta @ W[i].T)

    # Reverse to match the order of W
    gradients.reverse()

    return gradients


def plot_results(X, y, Z, losses):
    """
    Visualization function to plot training results

    INPUT:
        X - input features (first column is the actual feature)
        y - true labels
        Z - list of network outputs (Z[-1] contains final predictions)
        losses - array of loss values over training
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Predictions vs True values
    predictions = Z[-1].flatten()

    # Sort for smooth line plot
    sorted_indices = np.argsort(X)
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]
    pred_sorted = predictions[sorted_indices]

    ax1.scatter(X, y, alpha=0.5, label="True data", s=30)
    ax1.plot(
        X_sorted, pred_sorted, "r-", linewidth=2, label="Neural Network Prediction"
    )
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.set_title("Neural Network Fit", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training Loss
    ax2.plot(losses, linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("MSE Loss", fontsize=12)
    ax2.set_title("Training Loss Over Time", fontsize=14)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print final statistics
    final_loss = losses[-1]
    print(f"\nFinal MSE Loss: {final_loss:.6f}")
    print(f"Final RMSE: {np.sqrt(final_loss):.6f}")
