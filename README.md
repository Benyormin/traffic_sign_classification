# Traffic Sign Classification

This project aims to classify  Traffic Sign Classification

This project aims to classify traffic signs using deep learning techniques. It utilizes TensorFlow and a convolutional neural network (CNN) to train a model that can predict the type of traffic sign from input images.

## Dataset

The dataset used for this project can be found on Kaggle at the following link: [Traffic Signs Classification Dataset](https://www.kaggle.com/datasets/flo2607/traffic-signs-classification/). It consists of images of various traffic signs, organized into different directories based on their corresponding classes.

To use this dataset, you will need to download it from the provided link and place it in the appropriate directory.

## Preprocessing

The images in the dataset are preprocessed before training the model. The `preprocess` function reads the image file, decodes it, and normalizes the pixel values. Additionally, there is a `resize` function available to resize the images to a specific width while maintaining the aspect ratio.

## Model Architecture

The model architecture consists of a sequential stack of layers. The input images are flattened and passed through fully connected (dense) layers with ReLU activation functions. The final layer uses a softmax activation function to classify the images into one of the traffic sign classes.

## Training

The model is trained using the `fit` method of the `Sequential` model. The training data is split into training and test sets, with a certain percentage used for training and evaluation. The model is compiled with the Adam optimizer and the sparse categorical cross-entropy loss function. Accuracy is used as the evaluation metric.

## Usage

To use this project, follow these steps:

1. Download the Traffic Signs Classification dataset from Kaggle.
2. Place the dataset in the appropriate directory.
3. Install the required dependencies listed in `requirements.txt` run pip install -r requirements.txt.
5. Run the script to train the model and evaluate its performance.

## Results

After training the model, the results can be analyzed. The model's performance on the test set can be evaluated by calculating the accuracy metric. Additionally, you can use the trained model to make predictions on new images or real-time video data.

## Future Improvements

Here are some potential improvements that can be made to enhance the project:

- Data augmentation techniques to increase the size and diversity of the dataset.
- Hyperparameter tuning to optimize the model's performance.
- Visualizations of the model's predictions and learned features.
- Deployment of the model in a production environment.
