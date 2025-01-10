# Introduction to Machine Learning course - Code Assignment 1
# Authors: Tal Grossman, Amir Sharif Jamal, Din Carmon:
# Tal Grossman      201512282
# Amir Sharif Jamal 213850811
# Din Carmon        209325026

# Main file to run all parts

import os
import warnings

# Local imports
import get_data
import part1_neural_network
import part2_decision_trees

RANDOM_SEED = None
OUTPUT_DIR = "./results"

def run_part_1(samples_train, samples_test, labels_train, labels_test):
    part1_output_dir = os.path.join(OUTPUT_DIR, "part1")
    os.makedirs(part1_output_dir, exist_ok=True)
    part1_neural_network.train_and_eval(train_data=(samples_train, labels_train),
                                        test_data=(samples_test, labels_test),
                                        random_seed=RANDOM_SEED,
                                        output_dir=part1_output_dir)

def run_part_2(samples_train, samples_test, labels_train, labels_test):
    possible_splits = part2_decision_trees.build_all_possible_splits(samples_train)

    decision_tree = part2_decision_trees.build_tree(list(zip(samples_train, labels_train)),
                                                    possible_splits)

    print("Decision Tree:\n")
    print(decision_tree)

    pat2_output_dir = os.path.join(OUTPUT_DIR, "part2")
    os.makedirs(pat2_output_dir, exist_ok=True)
    with open(os.path.join(pat2_output_dir, "decision_tree.txt"), "w") as f:
        f.write(str(decision_tree))

    print(f"Accuracy of ID3 Constructed decision tree: {part2_decision_trees.compute_accuracy(decision_tree,
                                                                                              list(zip(samples_test,
                                                                                                       labels_test)))}")

def main():
    warnings.filterwarnings('error', category=RuntimeWarning)

    # Get the data

    samples_train, samples_test, labels_train, labels_test = get_data.get_dataset(desired_dataset="breast_cancer",
                                                                         testing_size=0.2,
                                                                         random_state=RANDOM_SEED)

    # Run part1
    run_part_1(samples_train, samples_test, labels_train, labels_test)

    # Run part 2
    run_part_2(samples_train, samples_test, labels_train, labels_test)

if __name__ == '__main__':
    main()


