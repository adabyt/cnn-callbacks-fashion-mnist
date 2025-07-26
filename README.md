# 👕 CNN with MobileNetV2 & Keras Callbacks on Fashion MNIST

This project demonstrates how to use **MobileNetV2** as a feature extractor for the **Fashion MNIST** dataset, combining **transfer learning** with a custom classification head.  
It also showcases the use of **Keras callbacks** — including **EarlyStopping**, **ModelCheckpoint**, and **TensorBoard** — to optimise training and track performance.

The model’s **best hyperparameters** (dense units, dropout, learning rate, optimiser) were sourced from prior hyperparameter tuning using **Random Search**, which can be found here: [cnn-keras-tuning-fashion-mnist](https://github.com/adabyt/cnn-keras-tuning-fashion-mnist).

---

## 🚀 Features
- Uses **MobileNetV2** pretrained on ImageNet for feature extraction  
- Adds a **custom dense layer & dropout** for Fashion MNIST classification  
- Implements **key Keras callbacks**:  
  - **EarlyStopping** → stops training when validation loss stops improving  
  - **ModelCheckpoint** → saves the best-performing model automatically  
  - **TensorBoard** → visualises training curves, histograms, and more  
- Evaluates model performance across **multiple independent runs** to assess stability

---

## 📂 Project Structure
```plaintext
cnn-callbacks-fashion-mnist/
│
├── callbacks_fashion_mnist.py   # Main training script
├── saved_models/                # Best model saved as .keras
├── logs/                        # TensorBoard logs
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠 Installation & Setup

1️⃣ **Clone the repo**
```bash
git clone https://github.com/adabyt/cnn-callbacks-fashion-mnist.git
cd cnn-callbacks-fashion-mnist
```

2️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

3️⃣ **Run the script**
```bash
python callbacks_fashion_mnist.py
```

4️⃣ **Launch TensorBoard (optional)**
```bash
tensorboard --logdir logs/fit
```
Then open [http://localhost:6006](http://localhost:6006) to visualise training.

---

## 📊 Model Summary
The final model structure:

- **MobileNetV2 (frozen)** → feature extraction  
- **Global Average Pooling**  
- **Dense (480 units, ReLU)**  
- **Dropout (0.3)**  
- **Dense (10 units, Softmax)**  

_Total trainable parameters: ~619k (MobileNetV2 frozen)_

---

## 📈 Results & Observations

### 🔍 Performance Across Runs
The model was trained multiple times to evaluate run-to-run consistency:

| Run | Test Accuracy | Test Loss | Epochs Used |
|----:|--------------:|----------:|------------:|
|  1  | 0.9079        | 0.2543    | 16          |
|  2  | 0.9076        | 0.2590    | 17          |
|  3  | 0.9056        | 0.2559    | 17          |
|  4  | 0.9080        | 0.2534    | 17          |
|  5  | 0.9065        | 0.2555    | 17          |
|  6  | 0.9052        | 0.2583    | 14          |

**Overall Performance:**  
**\`0.9068 ± 0.0011\`** after 6 runs.

---

### 📌 Why do results vary per run?

1. **Shuffle order** – Different batch orders change weight updates.  
2. **Weight initialisation** – Initial weights differ unless a seed is fixed.  
3. **Dropout randomness** – Different neurons drop each epoch.  
4. **Adam optimiser’s internal state** – Tracks moving averages of gradients, which evolve differently per run.  
5. **GPU non-determinism** (less likely for smaller models) – Minor floating-point differences can accumulate.

---

### 📉 Signs of Overfitting
The logs suggest mild overfitting: training accuracy improved faster than validation accuracy.

**Ways to mitigate overfitting:**
- Add more data  
- Use data augmentation  
- Apply regularization (L1/L2)  
- Increase dropout  
- Rely on early stopping  
- Add batch normalisation  
- Simplify model architecture  
- Tune hyperparameters further  

---

## 🏁 Conclusion

This project shows how:
- **Transfer learning with MobileNetV2** can effectively classify Fashion MNIST with minimal training time.  
- **Callbacks** like EarlyStopping and ModelCheckpoint prevent overtraining and save the best model automatically.  
- **TensorBoard** provides invaluable visualisation for debugging and performance tracking.  

**Final test accuracy: ~90.7%.**
