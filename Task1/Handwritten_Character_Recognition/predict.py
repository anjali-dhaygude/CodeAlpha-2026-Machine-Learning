import tensorflow as tf
import pandas as pd
import numpy as np
import zipfile
import os
import matplotlib.pyplot as plt


# ==========================================
# 1. YOUR CSV ZIP FILE
# ==========================================

ZIP_FILE = r"C:\Users\ranjit\Downloads\mnist_test.csv.zip"


# ==========================================
# 2. Load trained CNN model
# ==========================================

print("Loading trained model...")

model = tf.keras.models.load_model(
    "model/handwritten_model.keras"
)

print("Model loaded successfully!")


# ==========================================
# 3. Open ZIP file
# ==========================================

print("\nOpening CSV file...")

with zipfile.ZipFile(ZIP_FILE, "r") as zip_file:

    print("Files inside ZIP:")
    print(zip_file.namelist())

    # Get CSV file from ZIP
    csv_file = [
        file for file in zip_file.namelist()
        if file.endswith(".csv")
    ][0]

    print("\nReading:", csv_file)

    with zip_file.open(csv_file) as file:
        data = pd.read_csv(file)


# ==========================================
# 4. Display dataset information
# ==========================================

print("\nDataset shape:", data.shape)

print("\nFirst 5 rows:")
print(data.head())


# ==========================================
# 5. Separate label and pixels
# ==========================================

# Usually the first column is the digit label
y = data.iloc[:, 0].values

# Remaining 784 columns are image pixels
x = data.iloc[:, 1:].values


# ==========================================
# 6. Normalize pixels
# ==========================================

x = x / 255.0


# ==========================================
# 7. Convert to CNN format
# ==========================================

x = x.reshape(-1, 28, 28, 1)


# ==========================================
# 8. Select one image
# ==========================================

image_number = 0

image = x[image_number]

actual_digit = y[image_number]


# ==========================================
# 9. Predict digit
# ==========================================

prediction = model.predict(
    image.reshape(1, 28, 28, 1),
    verbose=0
)

predicted_digit = np.argmax(prediction)


# ==========================================
# 10. Display result
# ==========================================

print("\n==============================")
print("HANDWRITTEN CHARACTER RESULT")
print("==============================")

print("Actual digit    :", actual_digit)
print("Predicted digit :", predicted_digit)


# ==========================================
# 11. Display image
# ==========================================

plt.imshow(
    image.reshape(28, 28),
    cmap="gray"
)

plt.title(
    "Actual: " + str(actual_digit)
    + " | Predicted: " + str(predicted_digit)
)

plt.axis("off")
plt.show()