import tensorflow as tf
import os

# ==============================
# 1. Load MNIST training data
# ==============================

print("Loading MNIST dataset...")

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)


# ==============================
# 2. Normalize images
# ==============================

x_train = x_train / 255.0
x_test = x_test / 255.0


# ==============================
# 3. Reshape images
# ==============================

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# ==============================
# 4. Create CNN
# ==============================

model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])


# ==============================
# 5. Compile CNN
# ==============================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==============================
# 6. Train CNN
# ==============================

print("\nTraining CNN...")

model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.1
)


# ==============================
# 7. Test CNN
# ==============================

loss, accuracy = model.evaluate(
    x_test,
    y_test
)

print("\nTest Accuracy:", accuracy)


# ==============================
# 8. Create model folder
# ==============================

os.makedirs("model", exist_ok=True)


# ==============================
# 9. Save trained model
# ==============================

model.save("model/handwritten_model.keras")

print("\nModel saved successfully!")
print("Location:")
print("model/handwritten_model.keras")