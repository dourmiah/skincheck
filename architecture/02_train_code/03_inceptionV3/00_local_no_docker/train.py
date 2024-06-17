import tensorflow as tf
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

# Prétraitement des images avec ImageDataGenerator
k_Batch_Size = 32
k_Img_Height = 299  # 224
k_Img_Width = 299  # 224

k_data_dir = "../data_4/train"
k_nb_classes = 4
k_epochs = 15
k_steps_per_epoch = 20

os.makedirs("./img", exist_ok=True)

# C'est là qu'on fait le split entre train et validation.
# Voir le 0.3. Pas de data augmentation
img_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255, validation_split=0.3
)

# On crée 2 flux d'images : entrainement et validation
# train_data_dir = "../data/train"
train_data_dir = k_data_dir
img_generator_flow_train = img_generator.flow_from_directory(
    directory=train_data_dir,
    target_size=(k_Img_Height, k_Img_Width),
    batch_size=k_Batch_Size,
    shuffle=True,
    subset="training",
)

img_generator_flow_valid = img_generator.flow_from_directory(
    directory=train_data_dir,
    target_size=(k_Img_Height, k_Img_Width),
    batch_size=k_Batch_Size,
    shuffle=True,
    subset="validation",
)

num_classes = img_generator_flow_train.num_classes
print(f"Number of classes: {num_classes}")

# Utilisation du modèle pré-entraîné sans fine-tuning
base_model = tf.keras.applications.InceptionV3(
    input_shape=(k_Img_Height, k_Img_Width, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False

model = tf.keras.Sequential(
    [
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(
            # k_nb_classes classes à prédire
            k_nb_classes,
            activation="softmax",
        ),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
)

model.fit(
    img_generator_flow_train,
    validation_data=img_generator_flow_valid,
    steps_per_epoch=k_steps_per_epoch,
    epochs=k_epochs,
)

history_dict = model.history.history

plt.figure()
plt.plot(history_dict["categorical_accuracy"], c="r", label="Train Accuracy")
plt.plot(history_dict["val_categorical_accuracy"], c="b", label="Validation Accuracy")
plt.legend()
plt.title("Accuracy vs epochs")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"./img/{timestamp}_accuracy.png")

plt.figure()
plt.plot(history_dict["loss"], c="r", label="Train Loss")
plt.plot(history_dict["val_loss"], c="b", label="Validation Loss")
plt.legend()
plt.title("Loss vs epochs")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"./img/{timestamp}_loss.png")

# Évaluation du modèle sur les données de validation
# Pour img_generator_flow_valid voir la ligne
# img_generator_flow_valid = img_generator.flow_from_directory()
y_true = img_generator_flow_valid.classes
y_pred = np.argmax(model.predict(img_generator_flow_valid), axis=-1)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"./img/{timestamp}_confusion_matrix.png")


# Voir plus tard
# average=None => pour chaque classe
# recall = recall_score(y_true, y_pred, average=None)
# specificity = []
# for i in range(num_classes):
#     tn = np.sum(cm) - (np.sum(cm[:, i]) + np.sum(cm[i, :]) - cm[i, i])
#     fp = np.sum(cm[:, i]) - cm[i, i]
#     specificity.append(tn / (tn + fp))
#
# for i in range(num_classes):
#     print(f"Class {i}: Sensitivity (Recall) = {recall[i]}, Specificity = {specificity[i]}")
