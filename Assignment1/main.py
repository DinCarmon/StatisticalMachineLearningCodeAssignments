# Introduction to Machine Learning course - Code Assignment 1
# Authors: Tal Grossman, Amir Sharif Jamal, Din Carmon:
# Tal Grossman      201512282
# Amir Sharif Jamal 213850811
# Din Carmon        209325026

# main file to run all parts
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# local imports
import part1_neural_network 

RANDOM_SEED = 0
OUTPUT_DIR = "./results"


def get_dataset(desired_dataset="breast_cancer", testing_size=0.2, random_seed=42):
    """
    Get the dataset according to the desired dataset. 
    scale the data and split it to train and test sets.
    desired_dataset: str, the dataset to get.
    testing_size: float, the size of the test set.
    random_seed: int, the random seed.
    """
    scaler = StandardScaler()
    if desired_dataset == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        # Normalize the data
        X = scaler.fit_transform(data.data)
        Y = data.target.reshape(-1, 1)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=testing_size, random_state=random_seed)
    else:
        raise ValueError("only breast_cancer dataset is supported")
    return X_train, X_test, Y_train, Y_test


if __name__ == '__main__':
    # get the data
    X_train, X_test, Y_train, Y_test = get_dataset(
        desired_dataset="breast_cancer", testing_size=0.2, random_seed=RANDOM_SEED)

    # run part1
    part1_output_dir = os.path.join(OUTPUT_DIR, "part1")
    os.makedirs(part1_output_dir, exist_ok=True)
    part1_neural_network.train_and_eval(train_data=(X_train, Y_train),
                                 test_data=(X_test, Y_test),
                                 random_seed=RANDOM_SEED,
                                 output_dir=part1_output_dir)
