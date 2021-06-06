import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_data(dataset, img_size = 64):
    """
    Loads images from dataset and returns it in the desired format as numpy arrays

    Arguments:
    dataset -- dataset folder ('dev_set', 'test_set', or 'training_set')
    img_size (default: 64) -- size to resize images

    Returns:
    x_output -- numpy array of features (num_samples, img_size, img_size, 3)
    y_output -- numpy array of labels (1, num_samples)
    categories -- array of label names
    """

    np.random.seed(1)

    dirname = os.path.dirname(os.path.realpath('__file__'))
    
    categories = ['cat', 'dog']

    dataset_array = []

    for class_num in range(len(categories)):
        category = categories[class_num]
        path = os.path.join(dirname, 'dataset', dataset, category)

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
    y_output = y_output.reshape((1, y_output.shape[0])).astype(np.uint16)

    return x_output, y_output, categories