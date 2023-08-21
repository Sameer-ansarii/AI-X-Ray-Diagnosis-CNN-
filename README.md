# Data 

The dataset sourced from Kaggle (https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset) consists of Chest X-Ray images categorized into three classes: COVID-19, normal individuals, and viral pneumonia cases. The dataset is divided into two main directories, namely "train" and "test," each containing three subfolders corresponding to the three classes: "covid," "normal," and "viral pneumonia." In total, the dataset contains 317 image files, with 238 being in JPEG format, 68 in JPG format, and 11 in PNG format. This dataset serves as a valuable resource for training and evaluating Convolutional Neural Networks (CNNs) to accurately classify Chest X-Ray images into these three clinically relevant categories, which could contribute to the detection and diagnosis of respiratory diseases such as COVID-19 and viral pneumonia.

![dataset-cover](https://github.com/Sameer-ansarii/AI-X-Ray-Diagnosis-CNN-/assets/125865393/7a145124-f99a-4a3d-b604-1c3734a5637e)

# Problem Statement

Respiratory diseases, including COVID-19, viral pneumonia, and normal lung conditions, have significant implications for public health. The accurate and early detection of these conditions is crucial for effective medical intervention. This project focuses on leveraging Convolutional Neural Networks (CNNs) to classify Chest X-Ray images into three categories: COVID-19, normal, and viral pneumonia. The goal is to develop a robust model that aids medical professionals in making accurate and timely diagnoses. 
Respiratory diseases can be very serious and it is important to diagnose them quickly and accurately. Traditional methods of diagnosis can be slow and sometimes inaccurate. This project uses machine learning to develop a model that can automatically analyse chest X-ray images and classify them into three categories: COVID-19, normal, and viral pneumonia. The goal is to create a reliable model that can help doctors make accurate diagnoses more quickly. The project will also explore how different image formats affect the model's performance and how it could be used in real-world clinical settings.

# Teck Tech Used 

* Python (Programming Language)

* Pandas

* Numpy 

* Matplotlib 

* TensorFlow 

* Keras 

* ImageDataGenerator for Data Augmentation

* Sequential to build the structure of CNN

* Conv2D, MaxPooling2D, Flatten, Dense, Dropout

* Activation Functions: ReLU

* Callbacks for early stopping and model checkpoint

* Optimizers: Adam

* Loss Function: Categorical Cross Entropy

* Scikit-learn Metrics: Accuracy, Precision, Recall, F1 Score, Classification Report

* CV2

# Outcome

**Accurate Classification of Chest X-Ray Images for Respiratory Disease Diagnosis**

The CNN project using the provided chest X-ray image dataset has resulted in a strong model capable of accurately classifying images into three distinct categories: COVID-19, normal, and viral pneumonia. After rigorous training and evaluation, the model achieved an impressive overall accuracy of 88% on the test dataset, demonstrating its ability to effectively distinguish subtle patterns indicative of various respiratory conditions. The model's performance was consistently high across different image formats (JPEG, JPG, and PNG), demonstrating its adaptability and generalizability. This project highlights the importance of convolutional neural networks in medical image analysis and has the potential to help medical professionals accurately and promptly diagnose respiratory diseases, contributing to improved patient care and public health initiatives.
