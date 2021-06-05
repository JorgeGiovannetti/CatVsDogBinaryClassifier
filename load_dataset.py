import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_data(dataset):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    x_output -- numpy array of features ()
    """

    np.random.seed(1)

    dirname = os.path.dirname(os.path.realpath('__file__'))
    img_size = 100
    categories = ['cat', 'dog']

    dataset_array = []

    for category in categories:
        path = os.path.join(dirname, 'dataset', dataset, category)
        class_num = categories.index(category)

        for img in tqdm(os.listdir(path)):
            try:
                # Read image from file into array
                img_array = cv2.imread(
                    os.path.join(path, img), cv2.IMREAD_COLOR)

                # Resize image to (IMG_SIZE, IMG_SIZE)
                img_resized = cv2.resize(img_array, (img_size, img_size))
                dataset_array.append([img_resized, class_num])
            except Exception as e:
                pass

    output_array = np.array(dataset_array)

    # shuffle dataset
    np.random.shuffle(output_array)

    # Separate numpy array into features and labels
    x_output = np.array(output_array[:, 0])
    y_output = np.array(output_array[:, 1])

    return x_output, y_output, categories


# Load Data Demo
x, y, classes = load_data('dev_set')
print('data[0]', x[0], classes[y[0]])
