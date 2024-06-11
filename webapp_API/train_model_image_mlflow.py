import os
import numpy as np
import mlflow
import mlflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Définir les chemins
train_dir = 'data/train'
validation_dir = 'data/validation'

# Prétraitement des images
train_datagen = ImageDataGenerator(rescale=1.0/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(224, 224), batch_size=32, class_mode='binary')

# Charger le modèle pré-entraîné
base_model = MobileNetV2(weights='imagenet', include_top=False)

# Ajouter des couches personnalisées
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Créer le modèle complet
model = Model(inputs=base_model.input, outputs=predictions)

# Geler les couches du modèle de base
for layer in base_model.layers:
    layer.trainable = False

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Commencer une nouvelle exécution MLflow
mlflow.set_tracking_uri('https://mlflow-jedha-app-ac2b4eb7451e.herokuapp.com')  # Remplacez par l'URL de votre serveur MLflow
mlflow.set_experiment('Cat_Classifier_Experiment')

with mlflow.start_run():
    # Enregistrer les paramètres
    mlflow.log_param('learning_rate', 0.0001)
    mlflow.log_param('epochs', 10)
    mlflow.log_param('batch_size', 32)

    # Entraîner le modèle
    history = model.fit(train_generator, validation_data=validation_generator, epochs=10)

    # Enregistrer les métriques
    for epoch, acc in enumerate(history.history['accuracy']):
        mlflow.log_metric('train_accuracy', acc, step=epoch)
    for epoch, val_acc in enumerate(history.history['val_accuracy']):
        mlflow.log_metric('val_accuracy', val_acc, step=epoch)
    for epoch, loss in enumerate(history.history['loss']):
        mlflow.log_metric('train_loss', loss, step=epoch)
    for epoch, val_loss in enumerate(history.history['val_loss']):
        mlflow.log_metric('val_loss', val_loss, step=epoch)

    # Enregistrer le modèle
    mlflow.keras.log_model(model, "model")



print("Training completed and model saved to MLflow")
