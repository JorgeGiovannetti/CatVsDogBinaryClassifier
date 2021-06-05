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

    output_array = np.array(dataset_array, dtype=object)

    # shuffle dataset
    np.random.shuffle(output_array)

    # Separate numpy array into features and labels
    x_output = np.array(output_array[:, 0])
    y_output = np.array(output_array[:, 1])

    # Reshape arrays
    x_output = np.stack(x_output)
    y_output = y_output.reshape((1, y_output.shape[0]))

    return x_output, y_output, categories


# Load Data Demo
train_x_orig, train_y, classes = load_data('dev_set')

# Explore your dataset 
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]

print ("Number of training examples: " + str(m_train))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))