# Introduction to Machine Learning course - Code Assignment 1
# Authors: Tal Grossman, Amir Sharif Jamal, Din Carmon:
# Tal Grossman      201512282
# Amir Sharif Jamal 213850811
# Din Carmon        209325026

# Data retrieval

from typing import Any
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_dataset(desired_dataset : str = "breast_cancer",
                testing_size : float = 0.2,
                random_state : Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the dataset according to the desired dataset.
    scale the data and split it to train and test sets.
    desired_dataset: str, the dataset to get.
    testing_size: float, the fraction of the test set. A number between 0 and 1.
    random_state – Controls the shuffling applied to the data before applying the split.
                    Pass an int for reproducible output across multiple function calls.
                    None by default. Shall pass a random shuffle for each function call.
                    See :term:`Glossary <random_state>`.
    """
    scaler = StandardScaler()
    if desired_dataset == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        # Normalize the data
        samples : np.ndarray = scaler.fit_transform(data.data)
        labels : np.ndarray = data.target.reshape(-1, 1)

        samples_train, samples_test, labels_train, labels_test = (
            train_test_split(samples, labels, test_size = testing_size, random_state = random_state))
    else:
        raise ValueError("only breast_cancer dataset is supported")
    return samples_train, samples_test, labels_train.squeeze(), labels_test.squeeze()