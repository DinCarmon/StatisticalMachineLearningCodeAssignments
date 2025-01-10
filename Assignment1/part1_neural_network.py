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
from typing import Callable, List, Tuple

# Activation functions

def sigmoid(x : np.ndarray[int]):
    """
    Compute the sigmoid function, 1 / (1 + exp(-x)).
    Handles potential overflow issues by try and except.
    :param x - input, shape (values)
    """
    z : np.ndarray = np.zeros(x.shape)
    for i in range(len(x)):
        try:
            z[i] = 1 / (1 + np.exp(-x[i]))
        except RuntimeWarning:
            if x[i] > 0:
                z[i] = 0
            else:
                z[i] = 1

    return z

# Loss functions

def cross_entropy_loss(correct_label : np.ndarray,
                       estimated_label : np.ndarray) -> float:
    """
    :param correct_label: True labels, shape (samples, 1). The values should be 0 or 1.
    :param estimated_label: Predicted labels, shape (samples, 1). The values should be between 0 and 1.
    """
    # Add small epsilon to prevent log(0)
    epsilon = 1e-15

    # check is a smaller epsilon is needed
    smallest_non_zero = np.min(estimated_label[estimated_label > 0]) if np.any(estimated_label > 0) else None
    biggest_non_one = np.max(estimated_label[estimated_label < 1]) if np.any(estimated_label < 1) else None
    if (smallest_non_zero is not None and smallest_non_zero < epsilon) or \
        (biggest_non_one is not None and biggest_non_one > 1 - epsilon):
        # Choose an epsilon which is much more extreme than values which are estimated.
        epsilon = np.min([smallest_non_zero, 1 - biggest_non_one]) * 1e-5

    smallest_possible_epsilon = 1e-17
    if epsilon < smallest_possible_epsilon:
        # Python cannot handle such small values...
        epsilon = smallest_possible_epsilon

    estimated_label = np.clip(estimated_label, epsilon, 1 - epsilon)

    # Handle a divide by zero which occurs if epsilon is too small
    try:
        return -float(np.mean(correct_label * np.log(estimated_label) + (1 - correct_label) * np.log(1 - estimated_label)))
    except RuntimeWarning:
        return np.inf

def forward_pass(W : np.ndarray[int],
                 b : float,
                 X : np.ndarray[int, int],
                 activation_func : Callable = sigmoid) -> np.ndarray[int]:
    """
    Compute the forward pass of a 1 layer neural network with 1 final perceptron.
    :param W: Weights, shape: (features,).
    :param b: Bias
    :param X: Input features, shape: (samples, features)
    :param activation_func: Activation function (callable), default: sigmoid(x) = 1 / (1 + exp(-x))
    :return return A: Predictions, shape (samples, 1)
    """
    Z : np.ndarray = np.dot(X, np.transpose(W)) + b #
    A : np.ndarray = activation_func(Z)
    return A


def get_gradients(X : np.ndarray[int, int],
                  Y : np.ndarray[int],
                  Z : np.ndarray[int],
                  activation_func : Callable = sigmoid,
                  loss_function : Callable = cross_entropy_loss) -> Tuple[np.ndarray[int], float]:
    """
    Compute gradients for cross-entropy loss_function with sigmoid activation.
    :param X: input features, shape (samples, features)
    :param Y: true labels, shape (samples)
    :param Z: predictions, shape (samples)
    :param activation_func: Activation function
    :param loss_function: Loss function
    """
    if activation_func == sigmoid and loss_function == cross_entropy_loss:
        # Z lot of calculations yields the following gradient:
        # l' = -(y * log(x) + (1-y) * log(1-x)) = -(y/x + (y-1)/(1-x)) = - ( y(1-x) + (y-1)x ) / (x(1-x))=
        #       = (x - y) / (x(1-x))
        # sigmoid' = ... = sigmoid * (1 - sigmoid)
        # delta_1 = div_l(z) = (y - z) / (z(1-z))
        # for w_i: dZ / d(w_i) = l' * sigmoid' * x_i = (y-z) / (z(1-z)) * (z(1-z)) * x_i = (z - y) * x_i

        dZ : np.ndarray[Z.shape] = Z - Y  # Correct derivative for cross-entropy and sigmoid

        # We take the mean of the gradient across the computed gradient for each sample.
        dW : np.ndarray[X.shape[1]] = np.dot(X.T, dZ) / X.shape[0]
        db : float = float(np.mean(dZ))
    else:
        raise NotImplementedError(
            'Only sigmoid activation function and the cross entropy loss is supported')
    return dW, db

MAX_NUM_EPOCHS = 1000
LEARNING_PATIENCE = 0.9999
LEARNING_STOP_CRITERIA_NUM_OF_LAST_ROUNDS = 10
#STOPPING_CONDITION = 'num_epochs'
STOPPING_CONDITION = 'patience'

def single_layer_nn(train_data : Tuple[np.ndarray[int, int], np.ndarray[int]],
                    test_data : Tuple[np.ndarray[int, int], np.ndarray[int]],
                    W : np.ndarray[int],
                    b : float,
                    learning_rate : float = 0.1,
                    activation_func : Callable = sigmoid,
                    loss_func : Callable = cross_entropy_loss,
                    stopping_condition : str = 'num_epochs'):
    """
    Train a single layer neural network (with 1 output perceptron) using gradient descent
    :param train_data: tuple of samples (of shape [num of samples, num_of_coordinates]),
                                labels (of shape [num of samples])
    :param test_data: tuple of data_test, labels_test
    :param W: Weights: shape (num_of_coordinates)
    :param b: Bias : float
    :param learning_rate: Learning rate. The step taken at each iteration.
    :param activation_func: Activation function
    :param loss_func: Loss function
    :param stopping_condition: Stopping condition. Choose either "num_epochs" or "patience"
    """

    (data_train, labels_train) = train_data
    data_test, labels_test = test_data

    # Store metrics
    train_losses = []
    test_losses = []

    # Train the model
    loss_change_factor_last_rounds = np.ndarray(LEARNING_STOP_CRITERIA_NUM_OF_LAST_ROUNDS)
    loss_change_factor_last_rounds[:] = -np.inf

    num_of_epoch = 1
    while True:
        # Forward pass
        predictions = forward_pass(W, b, data_train
                        , activation_func)

        # Calculate the loss
        current_loss = loss_func(labels_train, predictions)
        train_losses.append(current_loss)

        # Backward pass
        dW, db = get_gradients(data_train
                    , labels_train, predictions,
                               activation_func = activation_func,
                               loss_function = loss_func)

        # # Update the weights and bias
        W -= learning_rate * dW
        b -= learning_rate * db

        test_loss = loss_func(labels_test, forward_pass(W, b, data_test, activation_func))
        test_losses.append(test_loss)

        if num_of_epoch > 1:
            loss_change_factor_last_rounds[:-1] = loss_change_factor_last_rounds[1:]
            loss_change_factor_last_rounds[-1] = test_losses[-1] / test_losses[-2]

        # Stopping criteria
        if stopping_condition == 'num_epochs':
            if num_of_epoch == MAX_NUM_EPOCHS:
                break
        elif stopping_condition == 'patience':
            if loss_change_factor_last_rounds.max() > LEARNING_PATIENCE:
                break
        else:
            raise ValueError('Invalid stopping condition. Choose either "num_epochs" or "patience"')

        num_of_epoch += 1

    return W, b, train_losses, test_losses

# hyper-parameters
EXPERIMENTS_LEARNING_RATES = [0.1, 0.01, 0.001]

def train(train_data : Tuple[np.ndarray[int, int], np.ndarray[int]],
          test_data : Tuple[np.ndarray[int, int], np.ndarray[int]],
          random_seed=42):
    """
    :param train_data: tuple of samples (of shape [num of samples, num_of_coordinates]),
                                labels (of shape [num of samples])
    :param test_data: tuple of samples (of shape [num of samples, num_of_coordinates]),
                                labels (of shape [num of samples])
    :param random_seed: Random seed. If not None, random generated values are reproducible,
                                        with different calls to this function.
    :return results_dict: {(learning_rate, initialization_type): (W, b, train_losses, test_losses, test_accuracy)}
    """

    data_train, labels_train = train_data
    data_test, labels_test = test_data
    num_features = data_train.shape[1]

    # Initialize weights and bias
    if random_seed is not None:
        np.random.seed(random_seed)
    # He initialization: references:
    # https://arxiv.org/abs/1502.01852
    # https://medium.com/@shauryagoel/kaiming-he-initialization-a8d9ed0b5899
    W_he_initialization = np.random.randn(num_features) * np.sqrt(2. /
                                                   num_features)  # He initialization
    b_he_initialization = np.zeros(1)  # Initialize bias to zero

    # normal initialization
    W_normal_initialization = np.random.randn(num_features)  # Normal initialization
    b_normal_initialization = np.zeros(1)  # Initialize bias to zero

    # initialization experiments
    init_exps = {
        'He Initialization': (W_he_initialization, b_he_initialization),
        'Normal Initialization': (W_normal_initialization, b_normal_initialization)
    }

    # list of all experiments as grid - i.e all combinations of learning rates and initializations
    all_experiments = [(lr, init) for lr in EXPERIMENTS_LEARNING_RATES for init in init_exps.items()]

    # results dict structure: {(learning_rate, initialization_type): (W, b, train_losses, test_losses, test_accuracy)}
    results_dict = {}
    # run the experiments
    for learning_rate, (initialization_type, (W_init, b_init)) in all_experiments:
        W = copy.deepcopy(W_init)  # deep copy to avoid changing the original weights saved in the dictionary
        b = copy.deepcopy(b_init)
        W_res, b_res, train_losses, test_losses = single_layer_nn(
            (data_train, labels_train),
            (data_test, labels_test),
            W, b,
            learning_rate = learning_rate,
            activation_func = sigmoid,
            loss_func = cross_entropy_loss,
            stopping_condition = STOPPING_CONDITION
        )
        test_accuracy = np.mean((forward_pass(W_res, b_res, data_test) > 0.5) == labels_test)
        results_dict[(learning_rate, initialization_type)] = (
            W_res, b_res, train_losses, test_losses, test_accuracy)

    return results_dict

PLOT_EVERY_N_EPOCHS = 10

def visualize_results(results_dict : dict,
                      output_dir="./results/part1",
                      show_graphs = False):
    """
    :param results_dict: {(learning_rate, initialization_type): (W, b, train_losses, test_losses, test_accuracy)}
                Expects all combinations of learning rates and initializations to be present in the dictionary.
    :param output_dir: Directory where the plots will be saved.
    """
    learning_rates = np.unique([run_configuration[0] for run_configuration in results_dict.keys()])
    initialization_types = np.unique([run_configuration[1] for run_configuration in results_dict.keys()])
    num_of_different_configurations_rates = len(results_dict.keys())

    # plot all experiments results
    fig, axs = plt.subplots(
        len(learning_rates), len(initialization_types), figsize=(15, 10))
    fig.suptitle('Experiments Results')
    for i, learning_rate in enumerate(learning_rates):
        for j, initialization_type in enumerate(initialization_types):
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
    if show_graphs:
        plt.show()
    # save plots
    experiments_plots_path = os.path.join(output_dir, "experiments_plots.png")
    fig.savefig(experiments_plots_path)

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
    if show_graphs:
        plt.show()
    # save plot
    best_model_plot_path = os.path.join(output_dir, "best_model_plot.png")
    fig.savefig(best_model_plot_path)
    # save the best W and b as json
    best_weights_path = os.path.join(output_dir, "best_weights.json")
    with open(best_weights_path, 'w') as f:
        json.dump({"W": best_W.tolist(), "b": best_b.tolist()}, f)

    print("Graphs + Best 1 layer NN configuration were saved to: " + output_dir)
    print(f"Best configuration 1 layer NN accuracy: {best_test_accuracy}\n")

    return results_dict