Pedestrian Gender Classification with the MIT Dataset: A Feature Fusion Approach

By Abdul Basit, AI/ML Engineer  

Published on May 12, 2025



Introduction

Pedestrian attribute recognition, such as gender classification, is a pivotal task in computer vision, with applications ranging from surveillance to retail analytics. Classifying gender from pedestrian images is challenging due to variations in pose, lighting, and background complexity. In this project, I developed a robust gender classification system using the MIT dataset, employing a feature fusion approach and Support Vector Machine (SVM) classification. This blog post outlines the methodology, implementation, and results, offering insights for practitioners tackling similar computer vision challenges.

Project Overview

The goal was to classify pedestrians as male or female using images from the MIT dataset, which contained 888 images (288 female, 600 male), presenting a class imbalance challenge. I addressed this through data augmentation and extracted a diverse set of features for classification.

Key Components





Dataset: MIT dataset with 888 images (288 female, 600 male).



Preprocessing: Image enhancement and data augmentation to balance classes.



Feature Extraction: Histogram of Oriented Gradients (HOG), Local Binary Patterns (LBP), Gray Level Co-occurrence Matrix (GLCM), and VGG19 deep features.



Feature Fusion: Concatenation of features followed by Principal Component Analysis (PCA) for dimensionality reduction.



Classification: Linear SVM with 10-fold cross-validation.



Environment: Google Colab with Python 3.11, OpenCV, scikit-image, TensorFlow, and scikit-learn.









































































Methodology

1. Data Preparation

The MIT dataset was stored in Google Drive under /content/drive/MyDrive/dip/MIT-IB/, with separate directories for female (female) and male (male1) images. I copied the images to a working directory (/content/drive/MyDrive/dataset/MIT/) for processing. The dataset was imbalanced, with 288 female and 600 male images.

To address the imbalance, I applied data augmentation:





Female images: Each image was augmented with 6 additional images (augmentation factor = 6), resulting in 288 × 7 = 2016 images.



Male images: Each image was augmented with 2 additional images (augmentation factor = 2), resulting in 600 × 3 = 1800 images.



Balancing: Randomly sampled 1800 female images to match the 1800 male images, yielding a balanced dataset of 3600 images.

Here’s the code snippet for the augmentation and balancing logic:

import random
import cv2
import numpy as np

def augment_data(img, label, augmentation_factor=5):
    augmented_images = [img]
    augmented_labels = [label]
    for _ in range(augmentation_factor):
        aug_img = img.copy()
        techniques = random.sample(['flip_horizontal', 'rotate_5', 'rotate_10', 'brightness', 'contrast', 'blur'], 2)
        for technique in techniques:
            if technique == 'flip_horizontal':
                aug_img = cv2.flip(aug_img, 1)
            elif technique == 'rotate_5':
                M = cv2.getRotationMatrix2D((aug_img.shape[1]//2, aug_img.shape[0]//2), 5, 1)
                aug_img = cv2.warpAffine(aug_img, M, (aug_img.shape[1], aug_img.shape[0]))
            elif technique == 'brightness':
                hsv = cv2.cvtColor(aug_img, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(hsv)
                factor = random.uniform(0.8, 1.2)
                v = cv2.multiply(v, factor)
                v = np.clip(v, 0, 255).astype(np.uint8)
                final_hsv = cv2.merge((h, s, v))
                aug_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
            # ... other techniques (contrast, blur, rotate_10)
        augmented_images.append(aug_img)
        augmented_labels.append(label)
    return augmented_images, augmented_labels

# In main()
X_augmented, y_augmented = [], []
for i, (img, label) in enumerate(zip(X_raw, y_raw)):
    aug_factor = 6 if label == 0 else 2  # Female: 6, Male: 2
    aug_images, aug_labels = augment_data(img, label, augmentation_factor=aug_factor)
    X_augmented.extend(aug_images)
    y_augmented.extend(aug_labels)

# Balance classes
from collections import Counter
target_count = 1800
X_balanced, y_balanced = [], []
for label in [0, 1]:
    indices = [i for i, y in enumerate(y_augmented) if y == label]
    selected_indices = random.sample(indices, target_count) if label == 0 else indices
    for idx in selected_indices:
        X_balanced.append(X_augmented[idx])
        y_balanced.append(y_augmented[idx])



2. Image Preprocessing

Images were resized to 128×256 pixels and converted to RGB format for consistency. To enhance image quality, I applied:





Histogram Equalization: Improved contrast in grayscale images.



Gaussian Blur: Reduced noise using a 5×5 kernel.



Adaptive Thresholding: Binarized images to emphasize key features.

def enhance_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img.copy()
    equalized = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    return thresh



3. Feature Extraction

To capture a comprehensive representation of the images, I extracted four types of features:





HOG: Encodes edge and gradient information, effective for pedestrian silhouettes.



LBP: Captures local texture patterns.



GLCM: Quantifies spatial relationships in pixel intensities.



VGG19: Extracts deep features from the FC7 layer of a pre-trained VGG19 model.

A notable challenge was ensuring compatibility with scikit-image’s hog function, as newer versions (0.19+) deprecated the multichannel parameter. Since HOG operates on grayscale images, I removed the parameter and ensured grayscale input:

from skimage.feature import hog
import cv2

def extract_hog_features(img, target_size=(128, 64)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img.copy()
    resized = cv2.resize(gray, target_size)
    features = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)
    return features



4. Feature Fusion and Dimensionality Reduction

I concatenated the HOG, LBP, GLCM, and VGG19 features into a single feature vector per image. To mitigate the curse of dimensionality, I applied PCA, retaining up to 500 components or 50% of the original dimensions, whichever was smaller:

import numpy as np
from sklearn.decomposition import PCA

def fuse_features(hog_features, lbp_features, glcm_features, vgg19_features):
    return np.hstack([hog_features, lbp_features, glcm_features, vgg19_features])

def apply_pca(features, n_components=500):
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)
    print(f"Explained variance with {n_components} components: {np.sum(pca.explained_variance_ratio_):.4f}")
    return features_pca, pca



5. Classification

A linear SVM classifier was trained on the PCA-reduced features, with an 80/20 train-test split. To evaluate robustness, I performed 10-fold cross-validation:

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def train_svm_classifier(X_train, y_train):
    svm = SVC(kernel='linear', C=1.0, probability=True)
    svm.fit(X_train, y_train)
    return svm

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    print(classification_report(y_test, y_pred, target_names=['Female', 'Male']))
    return metrics

Implementation Details

The pipeline was implemented in Python using Google Colab, with dependencies including OpenCV, scikit-image, TensorFlow, and scikit-learn. The code is modular, with functions for preprocessing, augmentation, feature extraction, fusion, and classification. The full code is available on https://github.com/theabdulbasitt/Pedestrian-Gender-Classification



Key implementation challenges included:





Class Imbalance: Resolved through targeted augmentation to balance the dataset.



High Dimensionality: Managed with PCA to prevent overfitting and reduce computational cost.



Library Compatibility: Addressed scikit-image’s hog function changes by removing the deprecated multichannel parameter.





Results

The pipeline processed 3600 images (1800 male, 1800 female) after augmentation and balancing. The SVM classifier achieved solid performance on the test set (720 images):

              precision    recall  f1-score   support
      Female       0.81      0.80      0.81       352
        Male       0.81      0.82      0.81       368
    accuracy                           0.81       720
   macro avg       0.81      0.81      0.81       720
weighted avg       0.81      0.81      0.81       720

To assess robustness, I performed 10-fold cross-validation. Below are the results for the first two folds, with similar performance observed across all folds:





Fold 1:

            precision    recall  f1-score   support
    Female       0.81      0.80      0.81       183
      Male       0.80      0.81      0.80       177
  accuracy                           0.81       360
 macro avg       0.81      0.81      0.81       360



Fold 2:

            precision    recall  f1-score   support
    Female       0.80      0.82      0.81       169
      Male       0.84      0.82      0.83       191
  accuracy                           0.82       360
 macro avg       0.82      0.82      0.82       360

The cross-validation results showed consistent performance, with mean accuracy around 0.81–0.82 across folds, indicating reliable generalization. Visualizations, including the confusion matrix and cross-validation plots, were generated and saved:

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
            xticklabels=['Female', 'Male'], yticklabels=['Female', 'Male'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('/content/drive/MyDrive/model/confusion_matrix.png')
plt.show()

Lessons Learned





Feature Diversity: Combining handcrafted (HOG, LBP, GLCM) and deep (VGG19) features enhanced classification performance by capturing complementary image characteristics.



Class Balancing: Strategic augmentation was critical to mitigate bias toward the majority class.



Library Updates: Adapting to library changes, such as scikit-image’s deprecated parameters, ensured code reliability.

Future Improvements





Advanced Augmentation: Incorporate generative models like GANs to create more diverse synthetic images.



Alternative Classifiers: Experiment with ensemble methods (e.g., Random Forest, XGBoost) or fine-tuned deep learning models.



Real-Time Deployment: Optimize the pipeline for real-time applications, such as surveillance systems.

Conclusion

This project demonstrates a robust approach to pedestrian gender classification using the MIT dataset. By addressing class imbalance, leveraging feature fusion, and employing PCA and SVM, the pipeline achieved a test accuracy of 81% and consistent cross-validation performance. I hope this post inspires you to explore computer vision projects and experiment with hybrid feature engineering techniques.

Connect with me on [LinkedIn/Email] or check out the code on [GitHub link]. Happy coding!

Keywords: Computer Vision, Gender Classification, MIT Dataset, Feature Fusion, SVM, Data Augmentation, PCA, VGG19.
