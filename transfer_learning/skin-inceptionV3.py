#!pip install pydot
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import shutil 
import json


print(tf.__version__)
path = tf.keras.utils.get_file(fname="/tf/notebooks/content/101_ObjectCategories.tar.gz",
                        origin="https://full-stack-bigdata-datasets.s3.eu-west-3.amazonaws.com/Deep+learning+Images+processing/Transfer+Learning/101_ObjectCategories.tar.gz")
 

# Full path of  the archive file 
filename = "/content/101_ObjectCategories.tar.gz"

# Target directory 
extract_dir = "/content"

# Format of archie file 
archive_format = "gztar"

# Unpack the archive file 
shutil.unpack_archive(filename, extract_dir, archive_format) 
print("Archive file unpacked successfully.") 

# Preprocessing with ImageDataGenerator
BATCH_SIZE = 32
image_height = 224
image_width = 224
img_generator = tf.keras.preprocessing.image.ImageDataGenerator(#rotation_range=90,
                                                                brightness_range=(0.5,1), 
                                                                #shear_range=0.2, 
                                                                #zoom_range=0.2,
                                                                channel_shift_range=0.2,
                                                                horizontal_flip=True,
                                                                vertical_flip=True,
                                                                rescale=1./255,
                                                                validation_split=0.3)
                                                                
#i#mg_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

img_generator_flow_train = img_generator.flow_from_directory(
    directory="/tf/notebooks/content/data/train",
    target_size=(image_height, image_width),
    batch_size=32,
    shuffle=True,
    subset="training")

img_generator_flow_valid = img_generator.flow_from_directory(
    directory="/tf/notebooks/content/data/train",
    target_size=(image_height, image_width),
    batch_size=32,
    shuffle=True,
    subset="validation")

# Fine-Tuning
base_model = tf.keras.applications.InceptionV3(input_shape=(image_height,image_width,3),
                                               include_top=False,
                                               weights = "imagenet"
                                               )
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 31
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(24, activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
              loss = tf.keras.losses.CategoricalCrossentropy(),
              metrics = [tf.keras.metrics.CategoricalAccuracy()])
model.fit(img_generator_flow_train, validation_data=img_generator_flow_valid, steps_per_epoch=20, epochs=50)

# Get the dictionary containing each metric and the loss for each epoch
history_dict = model.history.history

# Visualise train / Valid Accuracy
plt.plot(history_dict["categorical_accuracy"], c="r", label="train_accuracy")
plt.plot(history_dict["val_categorical_accuracy"], c="b", label="test_accuracy")

# Visualise train / Valid Loss
plt.plot(history_dict["loss"], c="r", label="train_loss")
plt.plot(history_dict["val_loss"], c="b", label="test_loss")

# Print what the top predicted class is
preds = model.predict(img)
pred_labels = tf.argmax(preds, axis = -1)

print("Prediction output:", preds)
print("Predicted label:", pred_labels)


'''Loop over each image, predicted label and heatmap in order to display the images with the superimposed grad cam heatmap and 
the corresponding predicted label. Do they match the true label? What happens to the grad cam for wrong predictions? 
Are there any grad cams that seem surprising to you?
'''
from pathlib import Path
import matplotlib
for img, pred_label, true_label, heatmap in zip(imgs, pred_labels, labels, heatmaps): 
  # We rescale heatmap to a range 0-255
  heatmap = np.uint8(255 * heatmap)

  # Display Grad CAM
  pred_file_path = np.argmax(img_generator_flow_valid.labels == pred_label)
  pred_label_name = Path(img_generator_flow_valid.filepaths[pred_file_path]).parent.name

  true_file_path = np.argmax(img_generator_flow_valid.labels == tf.argmax(true_label))
  true_label_name = Path(img_generator_flow_valid.filepaths[true_file_path]).parent.name

  print("Predicted label:",pred_label_name)
  print("True label:", true_label_name)
