import os
import tensorflow as tf
from tensorflow.keras import layers

# Data sourced from https://www.kaggle.com/puneet6060/intel-image-classification?
# Extract in the same directory as this file

train_dir = os.getcwd()+"/111880_269359_bundle_archive/seg_train/seg_train"
test_dir = os.getcwd()+"/111880_269359_bundle_archive/seg_test/seg_test"
pred_dir = os.getcwd()+"/111880_269359_bundle_archive/seg_pred/seg_pred"
num_classes = 6
batch_size = 32
img_size = (150, 150)
img_shape = img_size + (3,)
base_learning_rate = 0.0001



"""
try:
    base_model = tf.keras.models.load_model(os.getcwd()+'/modelTransfer')
except OSError:
    print("No model found")
    base_model = tf.keras.applications.Xception(input_shape=img_shape,
                                                include_top=False,
                                                weights='imagenet')  # Default is imagenet

else:
    print("Previous saved model loaded")
"""
# Load using TensorFlow Datasets: GPU
base_model = tf.keras.applications.Xception(input_shape=img_shape,
                                                include_top=False,
                                                weights='imagenet')  # Default is imagenet

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="training",
  seed=1,
  image_size=img_size,
  batch_size=batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="validation",
  seed=1,
  image_size=img_size,
  batch_size=batch_size)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_dir,
  seed=1,
  image_size=img_size,
  batch_size=batch_size)

print(str(len(train_ds.class_names)) + " categories loaded: ")
print(*train_ds.class_names)


# Data cache and prefetch for performance optimization
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(10).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

image_batch, label_batch = next(iter(train_ds))
feature_batch = base_model(image_batch)
base_model.trainable = True


# Fine-tune from this layer onwards
while True:
    print("Input layers to be frozen. Max layers: " + str(len(base_model.layers)))  # 60 at 32 dense .8763
    user_1 = input()
    try:
        for layer in base_model.layers[:int(user_1)]:
            layer.trainable = False
        break
    except TypeError:
        print("Input isn't an int")

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])
print("Starting model")
inputs = tf.keras.Input(shape=(None, None, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.xception.preprocess_input(x)

x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)  # https://stackoverflow.com/questions/49295311/what-is-the-difference-between-flatten-and-globalaveragepooling2d-in-keras
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(32, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
x = layers.Dense(num_classes)(x)
model = tf.keras.Model(inputs, x)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate/10),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=1
    )
base_model.save("modelTransfer")