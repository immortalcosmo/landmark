# Tuning using Keras Tuner
# Method 1, model build function
import tensorflow as tf
import kerastuner as kt
import os
import IPython

# Data sourced from https://www.kaggle.com/puneet6060/intel-image-classification?
# Extract in the same directory as this file

train_dir = os.getcwd()+"/111880_269359_bundle_archive/seg_train/seg_train"
test_dir = os.getcwd()+"/111880_269359_bundle_archive/seg_test/seg_test"
pred_dir = os.getcwd()+"/111880_269359_bundle_archive/seg_pred/seg_pred"
num_classes = 6
batch_size = 32
img_height, img_width = 150, 150

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


def build(hp):

    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
    ])
    for i in range(hp.Int('conv_blocks', 1, 3, default=1)):
        filters = hp.Int('filters_' + str(i), 32, 256, step=32)
        for _ in range(2):
            model.add(tf.keras.layers.Convolution2D(
                filters, kernel_size=(3, 3), padding='same'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.ReLU())
        if hp.Choice('pooling_' + str(i), ['avg', 'max']) == 'max':
            model.add(tf.keras.layers.MaxPool2D())
        else:
            model.add(tf.keras.layers.AvgPool2D())
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    hp_units = hp.Int('nodes', min_value=32, max_value=256, step=32)
    model.add(tf.keras.layers.Dense(units = hp_units, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(
        hp.Float('dropout', 0, 0.5, step=0.1, default=0.5)))
    model.add(tf.keras.layers.Dense(num_classes))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = hp_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


tuner = kt.Hyperband(
    build,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='my_dir',
    project_name='intro_to_kt',
    hyperband_iterations=2)


class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)


tuner.search(train_ds,
             validation_data=val_ds,
             epochs=10,
             callbacks=[ClearTrainingOutput()])  # tf.keras.callbacks.EarlyStopping(patience=1)
best_model = tuner.get_best_models(1)[0]
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

best_model.save("tuned_model")
print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

