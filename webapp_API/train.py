import os
import numpy as np
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
import json
from mlflow.models.signature import infer_signature

# Définir les chemins
train_dir = 'D:\Quentin\jedha\jedhaFullStack\docker\mlflow\package_serve\mlflow_project\data\train'
validation_dir = 'D:\Quentin\jedha\jedhaFullStack\docker\mlflow\package_serve\mlflow_project\data\test'
k_RelativePath = "SkinCheck"

# Prétraitement des images
img_generator = ImageDataGenerator()

directory_generator = img_generator.flow_from_directory(
    directory = "/home/app/data_sample/train", # the folder where the class subfolders can be found
    target_size = (512,512), # The (height,width) of the produced images
    class_mode = "sparse", # Wether the target should be represented by an index or a dummy vector
    batch_size=8, # The batch size of the produced batches
    shuffle = True #Whether to shuffle after all files have been selected once
    #subset = "training"
)

directory_generator_val = img_generator.flow_from_directory(
    directory = "/home/app/data_sample/test", # the folder where the class subfolders can be found
    target_size = (512,512), # The (height,width) of the produced images
    class_mode = "sparse", # Wether the target should be represented by an index or a dummy vector
    batch_size=8, # The batch size of the produced batches
    shuffle = True #Whether to shuffle after all files have been selected once
    #subset = "validation"
)

# Extraire les classes
classes = list(directory_generator_val.class_indices.keys())
num_classes = len(classes)  # Number of classes

imgs, targets = next(iter(directory_generator))

metrics=[SparseCategoricalAccuracy()])

# Commencer une nouvelle exécution MLflow
mlflow.set_tracking_uri('https://mlflow-jedha-app-ac2b4eb7451e.herokuapp.com')  # Remplacez par l'URL de votre serveur MLflow
mlflow.set_experiment('SkinCheck')

with mlflow.start_run():
    # Enregistrer les paramètres
    mlflow.log_param('learning_rate', 0.0001)
    mlflow.log_param('epochs', 25)
    mlflow.log_param('batch_size', 8)

    
    model = load_model("skincheck_cnn_sample_v10.keras")
    history = json.load(open('skincheckhistory_cnn_sample_v10.json', 'r'))


    # Enregistrer les métriques
    for epoch, acc in enumerate(history['sparse_categorical_accuracy']):
        mlflow.log_metric('train_accuracy', acc, step=epoch)
    for epoch, val_acc in enumerate(history['val_sparse_categorical_accuracy']):
        mlflow.log_metric('val_accuracy', val_acc, step=epoch)
    for epoch, loss in enumerate(history['loss']):
        mlflow.log_metric('train_loss', loss, step=epoch)
    for epoch, val_loss in enumerate(history['val_loss']):
        mlflow.log_metric('val_loss', val_loss, step=epoch)
        
     
    # Convert validation data to a format compatible with infer_signature
    val_imgs, val_targets = next(iter(directory_generator_val))
    val_imgs = np.array(val_imgs)  # Convertir en tableau NumPy
    

    predictions = model.predict(val_imgs)

    # Enregistrer le modèle
    signature = infer_signature(val_imgs, predictions)
    mlflow.keras.log_model(
            model=model,
            artifact_path=k_RelativePath,
            registered_model_name="keras_sequential",
            signature=signature,
        )
    
        # Enregistrer les classes comme un artefact
    with open("classes.json", "w") as f:
        json.dump(classes, f)
    mlflow.log_artifact("classes.json")



print("Training completed and model saved to MLflow")
