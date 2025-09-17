# 🐶🐱 Cat vs Dog Classifier (Transfer Learning with EfficientNetV2L)

## 📌 Project Overview

This project builds a **binary image classifier** (Cats 🐱 vs Dogs 🐶) using **Transfer Learning**.
We fine-tune a pre-trained **EfficientNetV2L** model (trained on ImageNet) and adapt it for our dataset.
The final model is trained **end-to-end** with additional dense layers for classification.

---
## Trained Model
The trained Cat vs Dog classifier model is hosted on Hugging Face:
👉 [Cat-Dog Classifier Model](https://huggingface.co/Abdulmoiz123/cat-dog-classifier)

---

## 🌐 Deployment

You can try the model live here:
👉 [**Live Demo Link**](https://your-deployed-project-link.com)

---


## 📂 Dataset

* Source: [Kaggle Cat & Dog Dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog)
* Train/Test split created using **`image_dataset_from_directory`**.
* Images resized to **224×224** for EfficientNet input.

## 🌀 Data Augmentation

Data Augmentation applied to reduce overfitting and improve generalization

* Random flip (horizontal/vertical)
* Random rotation
* Random zoom
* Random brightness/contrast adjustments

---

## 🏗️ Model Architecture

* **Base Model**: EfficientNetV2L (`imagenet` weights, `include_top=False`)
* **Fine-Tuning**: Last 20 layers unfrozen for task-specific learning
* **Custom Head**:

  * GlobalAveragePooling2D
  * Dense(512, ReLU) + Dropout(0.2)
  * Dense(512, ReLU) + Dropout(0.2)
  * Dense(1, Sigmoid) → Binary output (Dog = 1, Cat = 0)

---

## ⚙️ Training Details

* Optimizer: `Adam`
* Loss Function: `BinaryCrossentropy`
* Batch Size: 4
* Image Size: 224×224
* Early stopping + Model checkpoint used

---

## 📊 Evaluation Metrics

* **Accuracy**: Achieved high accuracy on validation set.
* **Loss Curve**: Shows convergence after fine-tuning.
* **Confusion Matrix**: Correctly separates cats vs dogs.

👉 Key Metrics:

* Training Accuracy: 0.9858
* Validation Accuracy: 0.9500
* Validation Loss: 0.0236
* Validation Precision: 0.9907
* Validation Recall: 1.0000


✅ These results show that the model generalizes very well, with high precision and recall, meaning it rarely misses cats/dogs and makes very few false positives.


---

## 🚀 Usage (Prediction on Raw Images)

```python
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("my_model.h5")

# Load and preprocess image
image = Image.open("test_image.jpg").resize((224,224))
img = np.asarray(image).astype("float32")
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Predict
pred = model.predict(img)
print("Prediction:", "Dog" if pred[0][0] > 0.5 else "Cat")
```

---

## 📌 Key Learnings

* Transfer learning dramatically reduces training time.
* Unfreezing last 20 layers allows the model to adapt to new dataset features.
* Preprocessing consistency (training vs raw prediction) is critical for good performance.
* Data Augmentation to reduce overfitting and improve generalization.
 







