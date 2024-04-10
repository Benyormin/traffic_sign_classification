# Traffic Sign Classification

This project aims to classify Traffic Sign Images

This project aims to classify traffic signs using deep learning techniques. It utilizes TensorFlow and a convolutional neural network (CNN) to train a model that can predict the type of traffic sign from input images.

## Dataset

The dataset used for this project is the [Traffic Signs Classification Dataset](https://www.kaggle.com/datasets/flo2607/traffic-signs-classification/) from [Kaggle](https://www.google.com/search?q=Kaggle). It contains a large collection of images representing various traffic signs, along with their corresponding labels.

## Model Architecture

The model architecture consists of a sequential stack of layers. The input images are flattened and passed through fully connected (dense) layers with ReLU activation functions. The final layer uses a softmax activation function to classify the images into one of the traffic sign classes.

## Preprocessing

The images in the dataset are preprocessed before training the model. The `preprocess` function reads the image file, decodes it, and normalizes the pixel values. Additionally, there is a `resize` function available to resize the images to a specific width while maintaining the aspect ratio.


## Training

The model is trained using the `fit` method of the `Sequential` model. The training data is split into training and test sets, with a certain percentage used for training and evaluation. The model is compiled with the Adam optimizer and the sparse categorical cross-entropy loss function. Accuracy is used as the evaluation metric.

## Installation

To run this project, please ensure that you have the following dependencies installed:

- [TensorFlow](https://www.google.com/search?q=TensorFlow)
- [Pandas](https://www.google.com/search?q=Pandas)
- [NumPy](https://www.google.com/search?q=NumPy)
- [Matplotlib](https://www.google.com/search?q=Matplotlib)
- [PIL](https://www.google.com/search?q=PIL) (Python Imaging Library)

You can install these dependencies by running the following command:
```
pip install -r requirements.txt
```

## Usage

1. Download the dataset from [here](https://www.kaggle.com/datasets/flo2607/traffic-signs-classification/) and extract it to a directory.
2. Update the `path` variable in the code to point to the directory where the dataset is stored.
3. Run the code to train and evaluate the model.

## Results

After training the model and evaluating it on the test dataset, the following results were obtained:

- Loss: 0.2064
- Accuracy: 94.29%

The high accuracy achieved indicates that the model is effective in classifying traffic signs. However, please note that the accuracy may vary depending on the dataset and other factors.

## Conclusion

In this project, we developed a deep learning model to classify traffic signs. The model achieved a high accuracy of 94.29% on the test dataset, demonstrating its effectiveness in recognizing and classifying different traffic signs.

Feel free to modify and experiment with the code to further improve the model's performance or adapt it to your specific needs.


## Future Improvements

Here are some potential improvements that can be made to enhance the project:

- Data augmentation techniques to increase the size and diversity of the dataset.
- Hyperparameter tuning to optimize the model's performance.
- Visualizations of the model's predictions and learned features.
- Deployment of the model in a production environment.
