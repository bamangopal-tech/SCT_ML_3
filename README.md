# SCT_ML_3
# SkillCraft Technology - Machine Learning Internship

## Task 03: Cats vs Dogs Classification using SVM

### Objective

Implement a Support Vector Machine (SVM) to classify images of cats
and dogs using the Kaggle Dogs vs Cats dataset.

## Dataset

The project uses the Kaggle Dogs vs Cats image dataset.

The dataset contains images belonging to two classes:

- Cat
- Dog

## Technologies Used

- Python
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- Scikit-image

## Machine Learning Algorithm

### Support Vector Machine (SVM)

Support Vector Machine is a supervised machine learning algorithm
used for classification and regression problems.

In this project, SVM is used to classify images into two classes:

- Cat
- Dog

An RBF kernel is used for classification.

## Feature Extraction

Raw images are resized to 64 × 64 pixels and converted to grayscale.

HOG (Histogram of Oriented Gradients) features are extracted from
the images before training the SVM model.

HOG captures important edge and shape information from the images.

## Project Workflow

1. Load the Cats and Dogs dataset.
2. Read the image files.
3. Assign labels to cats and dogs.
4. Resize the images.
5. Convert images to grayscale.
6. Extract HOG features.
7. Split the dataset into training and testing sets.
8. Train an SVM classifier.
9. Predict the test images.
10. Evaluate the model.
11. Display the confusion matrix.
12. Predict new images.

## Dataset Split

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

## Model Parameters

The SVM model uses:

- Kernel: RBF
- C: 10
- Gamma: Scale

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Results

The model classifies cat and dog images using HOG features and
Support Vector Machine.

The actual accuracy depends on the number of images used,
dataset preprocessing, and train-test split.

## Project Structure

```text
SCT_ML_Task03/
│
├── train/
│   ├── cat.0.jpg
│   ├── cat.1.jpg
│   ├── dog.0.jpg
│   ├── dog.1.jpg
│   └── ...
│
├── task3_svm.py
└── README.md
#PYTHON CODE
# ============================================================
# SkillCraft Technology - Machine Learning Internship
# Task 03: Cats vs Dogs Classification using SVM
# ============================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from skimage.feature import hog


# ------------------------------------------------------------
# 1. Dataset Path
# ------------------------------------------------------------

DATASET_PATH = "train"

# Image size
IMG_SIZE = 64

# Maximum number of images to use
# Set to None to use all images
MAX_IMAGES = 5000


# ------------------------------------------------------------
# 2. Load Images
# ------------------------------------------------------------

images = []
labels = []

files = os.listdir(DATASET_PATH)

# Keep only image files
image_files = [
    file for file in files
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]

# Limit dataset size if required
if MAX_IMAGES is not None:
    image_files = image_files[:MAX_IMAGES]

print("Number of images:", len(image_files))

for file in image_files:

    file_path = os.path.join(DATASET_PATH, file)

    # Determine label from filename
    if file.lower().startswith("cat"):
        label = 0
    elif file.lower().startswith("dog"):
        label = 1
    else:
        continue

    # Read image
    image = cv2.imread(file_path)

    if image is None:
        continue

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Extract HOG features
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    images.append(features)
    labels.append(label)


# ------------------------------------------------------------
# 3. Convert to NumPy Arrays
# ------------------------------------------------------------

X = np.array(images)
y = np.array(labels)

print("Feature shape:", X.shape)
print("Label shape:", y.shape)


# ------------------------------------------------------------
# 4. Split Dataset
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# 5. Create SVM Model
# ------------------------------------------------------------

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)


# ------------------------------------------------------------
# 6. Train Model
# ------------------------------------------------------------

print("\nTraining SVM model...")

model.fit(X_train, y_train)

print("Training completed.")


# ------------------------------------------------------------
# 7. Make Predictions
# ------------------------------------------------------------

y_pred = model.predict(X_test)


# ------------------------------------------------------------
# 8. Evaluate Model
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("MODEL PERFORMANCE")
print("===================================")

print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Cat", "Dog"]
    )
)


# ------------------------------------------------------------
# 9. Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(6, 5))

plt.imshow(cm, interpolation="nearest")

plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks([0, 1], ["Cat", "Dog"])
plt.yticks([0, 1], ["Cat", "Dog"])

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j],
                 ha="center",
                 va="center")

plt.show()


# ------------------------------------------------------------
# 10. Display Some Predictions
# ------------------------------------------------------------

plt.figure(figsize=(10, 8))

sample_indices = np.random.choice(
    len(X_test),
    min(9, len(X_test)),
    replace=False
)

for i, index in enumerate(sample_indices):

    prediction = y_pred[index]
    actual = y_test[index]

    plt.subplot(3, 3, i + 1)

    if prediction == 0:
        predicted_label = "Cat"
    else:
        predicted_label = "Dog"

    if actual == 0:
        actual_label = "Cat"
    else:
        actual_label = "Dog"

    plt.text(
        0.5,
        0.5,
        f"Actual: {actual_label}\nPredicted: {predicted_label}",
        ha="center",
        va="center",
        fontsize=12
    )

    plt.axis("off")

plt.suptitle("Sample Predictions")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 11. Predict a New Image
# ------------------------------------------------------------

def predict_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Image not found.")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_RGB2GRAY
    )

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    features = features.reshape(1, -1)

    prediction = model.predict(features)[0]

    if prediction == 0:
        result = "Cat"
    else:
        result = "Dog"

    print("\nPredicted Animal:", result)

    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.title("Prediction: " + result)
    plt.axis("off")
    plt.show()


# ------------------------------------------------------------
# Example:
# Uncomment the following line and provide an image path
# ------------------------------------------------------------

# predict_image("test_cat.jpg")
