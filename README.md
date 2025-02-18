Face Mask Detection
This project uses Convolutional Neural Networks (CNNs) to classify images of people as wearing a mask or not. It’s based on a dataset of labeled images, which is downloaded from Kaggle.

Features
Dataset: Contains images of people with and without masks.
Model: A CNN built using TensorFlow/Keras to classify mask usage.
Prediction: The model predicts if a person in an image is wearing a mask.
Installation
Install the required libraries:
bash
Copy
pip install tensorflow numpy matplotlib opencv-python scikit-learn kaggle pillow
Configure Kaggle API credentials to download the dataset:
bash
Copy
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
Usage
Download & Extract Dataset:
python
Copy
!kaggle datasets download -d omkargurav/face-mask-dataset
Preprocess Images: Resize images to 128x128 pixels and convert to NumPy arrays.

Train-Test Split: Split data into training and testing sets.

Model Building: Build a CNN with two convolutional layers, pooling, dropout, and dense layers.

Training & Evaluation: Train the model and evaluate accuracy.

Image Prediction: Input an image path to predict mask usage:

python
Copy
input_image = cv2.imread(input_image_path)
Results
Accuracy: The model achieves high accuracy on the test data.
Prediction Output: The model will print whether the person in the image is wearing a mask.
