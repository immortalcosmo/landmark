import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Data sourced from https://www.kaggle.com/puneet6060/intel-image-classification?
# Extract in the same directory as this file

train_dir = os.getcwd()+"/111880_269359_bundle_archive/seg_train/seg_train"
test_dir = os.getcwd()+"/111880_269359_bundle_archive/seg_test/seg_test"
pred_dir = os.getcwd()+"/111880_269359_bundle_archive/seg_pred/seg_pred"
num_classes = 6
batch_size = 32
img_height, img_width = 150, 150
try:
    model = tf.keras.models.load_model(os.getcwd()+'/model')
except:
    print("No model found")
else:
    print("Previous saved model loaded")

# Load using TensorFlow Datasets: GPU

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="training",
  seed=1,
  image_size=(img_height, img_width),
  batch_size=batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="validation",
  seed=1,
  image_size=(img_height, img_width),
  batch_size=batch_size)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_dir,
  seed=1,
  image_size=(img_height, img_width),
  batch_size=batch_size)

print(str(len(train_ds.class_names)) + " categories loaded: ")
print(*train_ds.class_names)

# Data cache and prefetch for performance optimization
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(10).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


def build():
    global model
    model = tf.keras.Sequential([
      layers.experimental.preprocessing.RandomFlip("horizontal"),
      layers.experimental.preprocessing.RandomRotation(0.1),
      layers.experimental.preprocessing.Rescaling(1./255),
      layers.Conv2D(16, 3, activation='relu', input_shape=(150, 150, 3)),
      layers.MaxPooling2D(2, 2),
      layers.Conv2D(32, 3, activation='relu'),
      layers.MaxPooling2D(2, 2),
      layers.Conv2D(64, 3, activation='relu'),
      layers.MaxPooling2D(2, 2),
      layers.Flatten(),
      layers.Dense(256, activation='relu'),
      layers.Dense(num_classes)
    ])

    # Train model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=3
    )

    model.save("model")


def evaluate():
    loss, acc = model.evaluate(test_ds)  # returns loss and metrics
    print("loss: %.2f" % loss)
    print("acc: %.2f" % acc)


def infer():
    img = tf.keras.preprocessing.image.load_img(
        test_dir + "/buildings/20057.jpg", target_size=(150, 150)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


infer()
