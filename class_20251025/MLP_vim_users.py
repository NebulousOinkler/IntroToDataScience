"""
Multilayer Perceptron Implementation
=====================================

In this project, you will implement a simple multilayer perceptron for a
regression problem.

EVALUATION:
-----------
This project must be successfully completed and submitted in order to receive
credit for this course. Your score on this project will be included in your
final grade calculation.

You are expected to write code where you see # YOUR CODE HERE within the
functions. Upon submitting your work, the code you write at these designated
positions will be assessed using an "autograder" that will run all test
functions to assess your code.

Be sure not to change the names of any provided functions, classes, or
variables, as this will interfere with the autograder.
"""

# =============================================================================
# SETUP AND IMPORTS
# =============================================================================

import numpy as np
from numpy.matlib import repmat
import sys
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time

from helper import *

print(f"You're running python {sys.version.split(' ')[0]}")


# =============================================================================
# DATA VISUALIZATION
# =============================================================================


def visualize_data():
    """
    Generate and visualize the 1-dimensional toy dataset.
    Note: X is of shape (N, 2) because we append 1 to each example for bias.
    """
    X, y = generate_data()

    print(
        f"The shape of X is {X.shape}. This is because we append 1 to "
        "each feature vector to introduce bias!"
    )

    plt.figure(figsize=(8, 6))
    plt.plot(X[:, 0], y, "*")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Training Data")
    plt.show()

    return X, y


# =============================================================================
# PART ZERO: ACTIVATION FUNCTIONS
# =============================================================================


def ReLU(z):
    """
    ReLU (Rectified Linear Unit) activation function.

    ReLU(z) = max(z, 0)

    INPUT:
        z - input array

    OUTPUT:
        result - ReLU applied element-wise to z
    """
    return np.maximum(z, 0)


def ReLU_grad(z):
    """
    Gradient of ReLU activation function.

    INPUT:
        z - input array

    OUTPUT:
        gradient - 1 where z > 0, else 0
    """
    return (z > 0).astype("float64")


def visualize_activation():
    """
    Visualize the ReLU activation function and its gradient.
    """
    z = np.linspace(-4, 4, 1000)

    plt.figure(figsize=(10, 5))
    plt.plot(z, ReLU(z), "b-", label="ReLU", linewidth=2)
    plt.plot(z, ReLU_grad(z), "r-", label="ReLU_grad", linewidth=2)
    plt.xlabel("z")
    plt.ylabel(r"$\max(z, 0)$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("ReLU Activation Function and Its Gradient")
    plt.show()


def demo_relu():
    """
    Display ReLU function and its gradient on a small example vector.
    """
    x = np.array([2.7, -0.5, -3.2])
    print("X:", x)
    print("ReLU(X):", ReLU(x))
    print("ReLU_grad(X):", ReLU_grad(x))


# =============================================================================
# PART ONE: WEIGHT INITIALIZATION [GRADED]
# =============================================================================


def initweights(specs):
    """
    Given a specification of the neural network, output a random weight array.

    INPUT:
        specs - array of length m+1. specs[0] should be the dimension of the
                feature and spec[-1] should be the dimension of output

    OUTPUT:
        W - array of length m, each element is a matrix
            where size(weights[i]) = (specs[i], specs[i+1])

    EXAMPLE:
        If specs = [2, 3, 1], this creates a network with:
        - 2 input features
        - 3 hidden units in the first hidden layer
        - 1 output unit

        W will be a list of 2 matrices:
        - W[0] has shape (2, 3)
        - W[1] has shape (3, 1)
    """
    # YOUR CODE HERE
    W = []
    return W


# =============================================================================
# PART TWO: FORWARD PASS [GRADED]
# =============================================================================


def forward_pass(W, xTr):
    """
    Forward pass through the neural network.

    INPUT:
        W - an array of L weight matrices
        xTr - nxd matrix. Each row is an input vector

    OUTPUTS:
        A - a list of matrices (of length L+1) that stores result of matrix
            multiplication at each layer
        Z - a list of matrices (of length L+1) that stores result of
            transition function at each layer

    ALGORITHM:
        For each layer i (from 0 to L-1):
            A[i+1] = Z[i] @ W[i]
            Z[i+1] = ReLU(A[i+1])  (except for the last layer)

        For the last layer:
            Z[L] = A[L]  (no activation function)
    """
    # YOUR CODE HERE
    # Initialize A and Z
    A = [xTr]
    Z = [xTr]

    return A, Z


# =============================================================================
# PART THREE: MSE LOSS [GRADED]
# =============================================================================


def MSE(out, y):
    """
    Mean Squared Error loss function.

    INPUT:
        out - output of network (n vector)
        y - training labels (n vector)

    OUTPUTS:
        loss - the MSE loss (a scalar)

    FORMULA:
        MSE = (1/n) * sum((out - y)^2)
    """
    # YOUR CODE HERE
    loss = 0.0

    return loss


def MSE_grad(out, y):
    """
    Gradient of MSE loss with respect to output.

    INPUT:
        out - output of network (n vector)
        y - training labels (n vector)

    OUTPUTS:
        grad - the gradient of the MSE loss with respect to out (n vector)

    FORMULA:
        ∇MSE = (2/n) * (out - y)
    """
    # YOUR CODE HERE
    n = len(y)
    grad = np.zeros(n)

    return grad


# =============================================================================
# PART FOUR: BACKPROPAGATION [GRADED]
# =============================================================================


def backprop(W, A, Z, y):
    """
    Backpropagation algorithm to compute gradients.

    INPUT:
        W - weights (list of matrices)
        A - output of forward pass before activation (list of matrices)
        Z - output of forward pass after activation (list of matrices)
        y - vector of size n (each entry is a label)

    OUTPUTS:
        gradients - the gradient with respect to W as a list of matrices

    ALGORITHM:
        1. Initialize delta = ∇MSE (gradient from loss function)
        2. For each layer (going backward):
            a. Compute gradient for weights: ∇W = Z_previous.T @ delta
            b. Update delta for previous layer: delta = ReLU_grad(A) * (delta @ W.T)
        3. Return gradients in forward order (same order as W)
    """
    # Convert delta to a column vector to make things easier
    delta = (MSE_grad(Z[-1].flatten(), y) * 1).reshape(-1, 1)

    # YOUR CODE HERE
    # compute gradient with backprop
    gradients = []

    return gradients


# =============================================================================
# TRAINING FUNCTION
# =============================================================================


def train_network(X, y, specs=[2, 20, 1], lr=0.001, epochs=100000, verbose=True):
    """
    Train a neural network using gradient descent.

    INPUT:
        X - training data (n x d matrix)
        y - training labels (n vector)
        specs - network architecture specification
        lr - learning rate
        epochs - number of training iterations
        verbose - whether to print progress

    OUTPUT:
        W - trained weights
        losses - array of loss values during training
    """
    # Initialize weights
    W = initweights(specs)

    # Track losses
    losses = np.zeros(epochs)

    # Training loop
    t0 = time.time()
    for i in range(epochs):
        # Forward pass
        A, Z = forward_pass(W, X)

        # Calculate loss
        losses[i] = MSE(Z[-1].flatten(), y)

        # Backward pass (compute gradients)
        gradients = backprop(W, A, Z, y)

        # Update weights
        for j in range(len(W)):
            W[j] -= lr * gradients[j]

        # Print progress
        if verbose and (i % 10000 == 0 or i == epochs - 1):
            print(f"Epoch {i}/{epochs}, Loss: {losses[i]:.6f}")

    t1 = time.time()

    if verbose:
        print(f"\nTraining completed in {t1-t0:.2f}s")
        print(f"Final loss: {losses[-1]:.6f}")

    return W, losses, A, Z


def plot_training_results(X, y, Z, losses):
    """
    Visualize training results.

    INPUT:
        X - training data (first column contains x values)
        y - true labels
        Z - network outputs (Z[-1] contains predictions)
        losses - array of loss values during training
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Predictions vs True values
    predictions = Z[-1].flatten()

    # Sort for smooth line plot
    sorted_indices = np.argsort(X[:, 0])
    X_sorted = X[sorted_indices, 0]
    y_sorted = y[sorted_indices]
    pred_sorted = predictions[sorted_indices]

    ax1.scatter(X[:, 0], y, alpha=0.5, label="True data", s=30)
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


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def test_initweights():
    """Test the initweights function."""
    print("\n" + "=" * 60)
    print("Testing initweights...")
    print("=" * 60)

    W = initweights([2, 3, 1])

    # Check that W is a list of appropriate length
    assert len(W) == 2, "W should have 2 weight matrices"

    # Check shapes (if W is not empty)
    if len(W) > 0 and len(W[0]) > 0:
        assert W[0].shape == (2, 3), f"W[0] should be (2, 3), got {W[0].shape}"
        assert W[1].shape == (3, 1), f"W[1] should be (3, 1), got {W[1].shape}"
        print("✓ initweights test passed!")
    else:
        print("⚠ initweights not yet implemented")


def test_forward_pass():
    """Test the forward_pass function."""
    print("\n" + "=" * 60)
    print("Testing forward_pass...")
    print("=" * 60)

    X, y = generate_data()
    W = initweights([2, 3, 1])

    try:
        A, Z = forward_pass(W, X)

        # Basic checks
        assert len(A) == 3, f"A should have 3 elements, got {len(A)}"
        assert len(Z) == 3, f"Z should have 3 elements, got {len(Z)}"

        if len(A[1]) > 0:
            n = X.shape[0]
            assert A[1].shape == (n, 3), f"A[1] should be ({n}, 3), got {A[1].shape}"
            assert Z[1].shape == (n, 3), f"Z[1] should be ({n}, 3), got {Z[1].shape}"
            print("✓ forward_pass test passed!")
        else:
            print("⚠ forward_pass not yet fully implemented")
    except Exception as e:
        print(f"✗ forward_pass test failed: {e}")


def test_MSE():
    """Test the MSE function."""
    print("\n" + "=" * 60)
    print("Testing MSE...")
    print("=" * 60)

    X, y = generate_data()
    W = initweights([2, 3, 1])
    A, Z = forward_pass(W, X)

    try:
        loss = MSE(Z[-1].flatten(), y)

        # Check that loss is a scalar
        assert np.isscalar(loss), "Loss should be a scalar"

        # Check that loss is non-negative
        assert loss >= 0, "Loss should be non-negative"

        print(f"✓ MSE test passed! Loss = {loss:.6f}")
    except Exception as e:
        print(f"✗ MSE test failed: {e}")


def test_MSE_grad():
    """Test the MSE_grad function."""
    print("\n" + "=" * 60)
    print("Testing MSE_grad...")
    print("=" * 60)

    X, y = generate_data()
    W = initweights([2, 3, 1])
    A, Z = forward_pass(W, X)

    try:
        grad = MSE_grad(Z[-1].flatten(), y)

        # Check shape
        n = X.shape[0]
        assert grad.shape == (n,), f"Gradient should be ({n},), got {grad.shape}"

        # Numerical gradient check
        out = np.array([1.0])
        y_test = np.array([1.2])
        numerical_grad = (MSE(out + 1e-7, y_test) - MSE(out - 1e-7, y_test)) / 2e-7
        analytical_grad = MSE_grad(out, y_test)

        if np.linalg.norm(numerical_grad - analytical_grad) < 1e-5:
            print("✓ MSE_grad test passed!")
        else:
            print("⚠ MSE_grad numerical check failed")
    except Exception as e:
        print(f"✗ MSE_grad test failed: {e}")


def test_backprop():
    """Test the backprop function."""
    print("\n" + "=" * 60)
    print("Testing backprop...")
    print("=" * 60)

    X, y = generate_data()
    W = initweights([2, 3, 1])
    A, Z = forward_pass(W, X)

    try:
        gradients = backprop(W, A, Z, y)

        # Check length
        assert len(gradients) == len(
            W
        ), f"Gradients should have same length as W ({len(W)}), got {len(gradients)}"

        # Check shapes
        if len(gradients) > 0 and len(gradients[0]) > 0:
            shapes_match = all(gradients[i].shape == W[i].shape for i in range(len(W)))
            if shapes_match:
                print("✓ backprop test passed!")
            else:
                print("⚠ backprop gradient shapes don't match weight shapes")
        else:
            print("⚠ backprop not yet fully implemented")
    except Exception as e:
        print(f"✗ backprop test failed: {e}")


def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 60)
    print("RUNNING ALL TESTS")
    print("=" * 60)

    test_initweights()
    test_forward_pass()
    test_MSE()
    test_MSE_grad()
    test_backprop()

    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """
    Main function to run the complete pipeline.

    This function demonstrates the full workflow:
    1. Visualize data
    2. Test individual components
    3. Train the network
    4. Visualize results
    """
    print("\n" + "=" * 60)
    print("MULTILAYER PERCEPTRON IMPLEMENTATION")
    print("=" * 60)

    # Step 1: Generate and visualize data
    print("\nStep 1: Generating data...")
    X, y = generate_data()
    print(f"Generated {len(y)} data points")

    # Step 2: Demonstrate ReLU
    print("\nStep 2: Demonstrating ReLU activation...")
    demo_relu()

    # Step 3: Run tests
    print("\nStep 3: Running tests...")
    run_all_tests()

    # Step 4: Train network (only if functions are implemented)
    print("\nStep 4: Training network...")
    try:
        W, losses, A, Z = train_network(
            X,
            y,
            specs=[2, 20, 1],  # 2 input, 20 hidden, 1 output
            lr=0.001,  # learning rate
            epochs=100000,  # number of epochs
            verbose=True,
        )

        # Step 5: Visualize results
        print("\nStep 5: Visualizing results...")
        plot_training_results(X, y, Z, losses)

    except Exception as e:
        print(f"\n⚠ Training not completed: {e}")
        print("This is likely because some functions are not yet implemented.")
        print("Complete the functions marked with '# YOUR CODE HERE' and try again!")


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    This block runs when the script is executed directly.

    To use this script:
    1. Implement the functions marked with '# YOUR CODE HERE'
    2. Run the script: python MLP_Class.py
    3. Or import and use individual functions in an interactive session
    """

    # Uncomment the line below to run the full pipeline
    # main()

    # Or run individual components:
    print("\nTo run the full pipeline, uncomment 'main()' at the bottom of this file.")
    print("\nOr you can:")
    print("  - Import this module: from MLP_Class import *")
    print("  - Run tests: run_all_tests()")
    print("  - Train network: W, losses, A, Z = train_network(X, y)")
    print("  - Visualize: plot_training_results(X, y, Z, losses)")
