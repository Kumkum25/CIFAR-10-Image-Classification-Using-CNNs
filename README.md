# 🖼️ CIFAR-10 Image Classification using CNN

## 📖 Overview
This project focuses on building and evaluating **Convolutional Neural Networks (CNNs)** for image classification using the **CIFAR-10 dataset**, which contains 60,000 color images across 10 classes such as airplanes, cars, birds, cats, and ships.  

The objective is to train CNN models that can accurately classify unseen images while addressing challenges like **overfitting** and **class imbalance**.

---

## 📊 Dataset Description
The **CIFAR-10** dataset consists of:
- **Training images:** 50,000  
- **Test images:** 10,000  
- **Image size:** 32×32 pixels  
- **Channels:** 3 (RGB)  
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

---

## ⚙️ Model Development
Multiple CNN architectures and optimizers were tested:
- **Small CNN (baseline)**
- **Custom CNN (deeper architecture)**
- **Optimizers:** Adam, SGD, RMSProp

Each model was trained using **categorical cross-entropy** loss and **softmax** activation in the final layer.

### 🔧 Regularization & Improvements
To address overfitting and improve generalization:
- **Data Augmentation:** random rotations, flips, and shifts  
- **Dropout layers:** to reduce reliance on specific neurons  
- **Early Stopping:** monitoring validation loss to avoid over-training  
- **Batch Normalization:** to stabilize and accelerate training  

---

## 🧠 Model Performance

| Model | Train Accuracy | Validation Accuracy | Test Accuracy | Train Loss | Validation Loss | Test Loss |
|--------|----------------|---------------------|----------------|-------------|------------------|------------|
| Small CNN | 0.6216 | 0.6752 | 0.6693 | 1.0797 | 0.9277 | 0.9439 |
| Adam | 0.6254 | 0.6771 | 0.6723 | 1.0733 | 0.9147 | 0.9309 |
| SGD | 0.5878 | 0.6391 | 0.6343 | 1.1793 | 1.0350 | 1.0490 |
| RMSprop | 0.6208 | 0.6833 | 0.6828 | 1.1088 | 0.9284 | 0.9367 |
| **Custom CNN** | **0.9226** | **0.8385** | **0.8382** | **0.2212** | **0.5339** | **0.5787** |

---

## 🔍 Analysis of Results

### Evidence of Overfitting
- The **Custom CNN** achieved very high training accuracy (**92.26%**) but lower validation (**83.85%**) and test accuracy (**83.82%**).  
- This indicates **overfitting**, as the model memorizes training patterns but doesn’t generalize perfectly to unseen data.

Despite this, the **Custom CNN** significantly outperformed all baseline models, showing it learned strong feature representations.

---

## 🧩 Observations
- The CNN model successfully learned **meaningful spatial features** from images.  
- **Data augmentation** and **dropout regularization** effectively reduced overfitting.  
- Validation accuracy near **73–84%** demonstrates solid generalization performance.  
- Minor **fluctuations in validation loss** are expected due to real-time augmentation and stochastic gradient updates.  

---

## 🧰 Technologies Used
- **Python**
- **TensorFlow / Keras**
- **NumPy, Pandas**
- **Matplotlib / Seaborn** (for visualization)
- **scikit-learn** (for evaluation metrics)

---

## 🚀 Key Insights
- CNNs effectively capture local spatial hierarchies in image data.  
- **Custom CNNs**, when paired with proper regularization, can outperform prebuilt baselines.  
- **Data augmentation** remains one of the simplest yet most effective techniques to improve model robustness.

---

## 🏁 Conclusion
The final **Custom CNN** model achieved:
- **Training Accuracy:** 92.26%  
- **Validation Accuracy:** 83.85%  
- **Test Accuracy:** 83.82%

✅ The model demonstrates strong generalization and classification capability on CIFAR-10.  
🔧 Further improvements could include fine-tuning architecture depth, using **transfer learning** with pretrained models like **ResNet or VGG16**, or applying **learning rate scheduling** to enhance convergence.

---

## 👩‍💻 Author
**Kumkum Kaushik**  
💼 Passionate about Deep Learning, Computer Vision, and building intelligent models that learn from data.
