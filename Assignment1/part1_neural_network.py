# Introduction to Machine Learning course - Code Assignment 1
# Authors: Tal Grossman, Amir Sharif Jamal, Din Carmon:
# Tal Grossman      201512282
# Amir Sharif Jamal 213850811
# Din Carmon        209325026

# Part 1: Single-Layer Neural Network with Gradient Descent

import os
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.format import EXPECTED_KEYS


# Activation functions


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Loss function


def cross_entropy_loss(Y, X):
    # Add small epsilon to prevent log(0)
    epsilon = 1e-15
    X = np.clip(X, epsilon, 1 - epsilon)
    return -np.mean(Y * np.log(X) + (1 - Y) * np.log(1 - X))


def inference(W, b, X, activation_func=sigmoid):
    """
    Compute the forward pass of the neural network.
    W: Weights, shape (features,). we reshape it to (features, 1)
    b: Bias, shape (1,)
    X: Input features, shape (samples, features)
    activation_func: Activation function
    return A: Predictions, shape (samples, 1)
    """
    Z = np.dot(X, W.reshape(-1, 1)) + b
    A = activation_func(Z)
    return A


def get_gradients(X, Y, A,
                  activation_func=sigmoid,
                  loss=cross_entropy_loss):
    """
    Compute gradients for cross-entropy loss with sigmoid activation.
    X: input features, shape (samples, features)
    Y: true labels, shape (samples, 1)
    A: predictions (after sigmoid), shape (samples, 1)
    """
    if activation_func == sigmoid and loss == cross_entropy_loss:
        dZ = A - Y  # Correct derivative for cross-entropy and sigmoid
        dW = np.dot(X.T, dZ) / X.shape[0]
        db = np.mean(dZ)
    else:
        raise NotImplementedError(
            'Only sigmoid activation function is supported')
    return dW, db


def single_layer_nn(train_data, test_data,
                    W, b,
                    learning_rate=0.1, num_epochs=1000,
                    learning_patience=None,
                    activation_func=sigmoid, loss_func=cross_entropy_loss):
    """
    Train a single layer neural network using gradient descent
    :param train_data: tuple of X_train, Y_train
    :param test_data: tuple of X_test, Y_test
    :param W: Weights
    :param b: Bias
    :param learning_rate: Learning rate
    :param num_epochs: MAX Number of epochs
    :param activation_func: Activation function
    :param loss: Loss function
    """

    X_train, Y_train = train_data
    X_test, Y_test = test_data

    # Store metrics
    train_losses = []
    test_losses = []

    not_improved_counter = 0
    best_test_loss = np.inf

    # Train the model
    for i in range(num_epochs):
        # Forward pass
        A = inference(W, b, X_train, activation_func)

        # Calculate the loss
        current_loss = loss_func(Y_train, A)
        train_losses.append(current_loss)

        # Backward pass
        dW, db = get_gradients(X_train, Y_train, A,
                               activation_func, loss=loss_func)

        # # Update the weights and bias
        W -= learning_rate * dW.squeeze()
        b -= learning_rate * db

        test_loss = loss_func(Y_test, inference(W, b, X_test, activation_func))
        test_losses.append(test_loss)

        if learning_patience is not None:
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                not_improved_counter = 0
            else:
                not_improved_counter += 1
                if not_improved_counter >= learning_patience:
                    break

    return W, b, train_losses, test_losses


def train_and_eval(train_data, test_data, random_seed=42,
                   output_dir="./results/part1"):
    """
    :param train_data: tuple of X_train(n_samples, m_featues), Y_train(n_samples, 1)
    :param test_data: tuple of X_test(n_samples, m_featues), Y_test(n_samples, 1)
    :param random_seed: Random seed
    return results_dict: {(learning_rate, initialization_type): (W, b, train_losses, test_losses, test_accuracy)}
    """

    # hyper-parameters
    MAX_NUM_EPOCHS = 1000
    LEARNING_PATIENCE = 100
    EXPERIMENTS_LEARNING_RATES = [0.1, 0.01, 0.001]
    PLOT_EVERY_N_EPOCHS = 10

    X_train, Y_train = train_data
    X_test, Y_test = test_data
    num_features = X_train.shape[1]

    # Initialize weights and bias
    np.random.seed(random_seed)
    # He initialization: references:
    # https://arxiv.org/abs/1502.01852
    # https://medium.com/@shauryagoel/kaiming-he-initialization-a8d9ed0b5899
    W_he = np.random.randn(num_features) * np.sqrt(2. /
                                                   num_features)  # He initialization
    b_he = np.zeros(1)  # Initialize bias to zero

    # normal initialization
    W_normal = np.random.randn(num_features)  # Normal initialization
    b_normal = np.zeros(1)  # Initialize bias to zero

    # initialization experiments
    init_exps = {
        'He Initialization': (W_he, b_he),
        'Normal Initialization': (W_normal, b_normal)
    }

    # list of all experiments as grid - i.e all combinations of learning rates and initializations
    all_experiments = [(lr, init) for lr in EXPERIMENTS_LEARNING_RATES for init in init_exps.items()]

    # results dict structure: {(learning_rate, initialization_type): (W, b, train_losses, test_losses, test_accuracy)}
    results_dict = {} 
    # run the experiments
    for learning_rate, (initialization_type, (W_init, b_init)) in all_experiments:
        W = copy.deepcopy(W_init) # deep copy to avoid changing the original weights saved in the dictionary
        b = copy.deepcopy(b_init) 
        W_res, b_res, train_losses, test_losses = single_layer_nn(
            (X_train, Y_train),
            (X_test, Y_test),
            W, b,
            learning_rate=learning_rate,
            num_epochs=MAX_NUM_EPOCHS,
            learning_patience=LEARNING_PATIENCE,
            activation_func=sigmoid,
            loss_func=cross_entropy_loss
        )
        test_accuracy = np.mean((inference(W_res, b_res, X_test) > 0.5) == Y_test)
        results_dict[(learning_rate, initialization_type)] = (
            W_res, b_res, train_losses, test_losses, test_accuracy)

    # plot all experiments results
    fig, axs = plt.subplots(
        len(EXPERIMENTS_LEARNING_RATES), len(init_exps), figsize=(15, 10))
    fig.suptitle('Experiments Results')
    for i, learning_rate in enumerate(EXPERIMENTS_LEARNING_RATES):
        for j, initialization_type in enumerate(init_exps.keys()):
            W, b, train_losses, test_losses, test_accuracy = results_dict[(
                learning_rate, initialization_type)]
            final_train_loss = train_losses[-1]
            final_test_loss = test_losses[-1]
            axs[i, j].plot(range(0, len(train_losses), PLOT_EVERY_N_EPOCHS),
                           train_losses[::PLOT_EVERY_N_EPOCHS], label=f"Training Loss: {final_train_loss:.3f}", color='b')
            axs[i, j].plot(range(0, len(test_losses), PLOT_EVERY_N_EPOCHS),
                           test_losses[::PLOT_EVERY_N_EPOCHS], label=f"Test Loss: {final_test_loss:.3f}", color='r')
            axs[i, j].set_title(
                f'Learning Rate: {learning_rate}, Initialization: {initialization_type}')
            axs[i, j].set_xlabel('Epochs')
            axs[i, j].set_ylabel('Loss')
            axs[i, j].legend(loc='upper right', title=f'Test Accuracy: {test_accuracy:.3f}')
            
    plt.tight_layout()
    plt.show()
    # save plots
    expiriments_plots_path = os.path.join(output_dir, "experiments_plots.png")
    fig.savefig(expiriments_plots_path)

    # plot the best model by taking the model with the final lowest test loss
    best_model = min(results_dict.items(), key=lambda x: x[1][3][-1])
    best_W, best_b = best_model[1][:2]
    best_test_accuracy = best_model[1][4]
    # plot the best model
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(0, len(best_model[1][2]), PLOT_EVERY_N_EPOCHS),
            best_model[1][2][::PLOT_EVERY_N_EPOCHS], label=f"Training Loss: {best_model[1][2][-1]:0.2}", color='b')
    ax.plot(range(0, len(best_model[1][3]), PLOT_EVERY_N_EPOCHS),
            best_model[1][3][::PLOT_EVERY_N_EPOCHS], label=f"Test Loss: {best_model[1][3][-1]:0.2}", color='r')
    ax.set_title(
        f'Best Model: Learning Rate: {best_model[0][0]}, Initialization: {best_model[0][1]}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right', title=f'Test Accuracy: {best_test_accuracy:.3f}')
    plt.tight_layout()
    plt.show()
    # save plot
    best_model_plot_path = os.path.join(output_dir, "best_model_plot.png")
    fig.savefig(best_model_plot_path)
    # save the best W and b as json
    best_weights_path = os.path.join(output_dir, "best_weights.json")
    with open(best_weights_path, 'w') as f:
        json.dump({"W": best_W.tolist(), "b": best_b.tolist()}, f)



    return results_dict