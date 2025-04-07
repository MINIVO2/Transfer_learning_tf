
# 🔧 Transfer Learning with TensorFlow - Part 2: Fine-Tuning

This notebook builds on the previous transfer learning project by adding **fine-tuning** to the workflow. We take a pre-trained model (EfficientNetB0) and unfreeze the top layers to retrain them on our custom dataset of food images.

---

## 📂 Dataset

We continue using the **10 Food Classes - 10%** dataset with:

- 📁 Training Data: 750 images
- 📁 Test Data: 2500 images

Same categories: pizza, steak, sushi, chicken curry, etc.

Dataset Link: [10 Food Classes - 10 Percent](https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip)

---

## 🧠 What is Fine-Tuning?

Fine-tuning means:
1. Start with a **pre-trained model** (e.g., EfficientNetB0).
2. **Unfreeze** the top layers.
3. Retrain them on your custom dataset to improve performance.

---

## 🔧 Steps Covered

### 1. Load and Preprocess Data
- Use `ImageDataGenerator` to prepare training and test data.

### 2. Load Pretrained Model (Feature Extractor)
- Use `EfficientNetB0` from **TensorFlow Hub**
- Initially, set `trainable=False` to extract features

### 3. Compile and Train (Feature Extraction)
- Train the model on the new dataset
- Evaluate and visualize results

### 4. Fine-Tune Model
- Unfreeze the **top layers**
- Recompile with a lower learning rate
- Continue training (fine-tune stage)

```python
base_model.trainable = True
fine_tune_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    ...
)
```

### 5. Evaluate and Visualize
- Plot training history for both phases
- Use TensorBoard for deeper analysis

---

## 🧰 Libraries Used

- `tensorflow`
- `tensorflow_hub`
- `matplotlib`
- `numpy`
- `os`, `datetime` (for logging)

---

## 📊 Results

- EfficientNetB0 achieved **higher accuracy** after fine-tuning
- Fine-tuning helps improve model understanding on small datasets

---

## 📁 Folder Structure

```
├── 10_food_classes_10_percent/
│   ├── train/
│   └── test/
├── tensorboard_logs/
├── 05_transfer_learning_tensorflow_part_2_fine_tuning.ipynb
├── README.md
```

---

## 🧪 Possible Improvements

- Try fine-tuning other models (e.g., ResNet, MobileNet)
- Use data augmentation
- Train on the full dataset

---

## 🙌 Credits

This project is part of the **TensorFlow Developer Course** by [Daniel Bourke](https://github.com/mrdbourke).

```

---

Let me know if you'd like a combined README for both notebooks, or a downloadable `.md` file!
