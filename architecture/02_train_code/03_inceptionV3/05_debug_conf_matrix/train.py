# Run this script  within a virtual env where tensorflow is available

# It runs locally InceptionV3 (with no tuning)
# Use to debug weird behavior in the confusion matrix display

import os
import random
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

k_data_dir = "../../../data_4/train"
k_nb_classes = 4
k_Batch_Size = 32
k_epochs = 2
k_steps_per_epoch = 20
k_LearningRate = 0.001
k_Img_Height = 299  # 224
k_Img_Width = 299  # 224

# Suite création de pred_n_compare
# ajout de model.trainable = False
# ajout de shuffle=False dans :
# img_generator_flow_validation = img_generator.flow_from_directory(
#         directory=train_data_dir,
#         target_size=(k_Img_Height, k_Img_Width),
#         batch_size=k_Batch_Size,
#         # shuffle=True,
#         shuffle=False,
#         subset="validation",
#     )
# J'invoque evaluate_model_1 et evaluate_model_2
# On obtient les mêmes matrices mais bon c'est pas top et shuffle=False

# shuffle=True : Les données sont mélangées avant d'être passées au modèle. Utilisé en entraînement pour améliorer
# la généralisation du modèle en exposant le modèle à des variations des données dans un ordre aléatoire.
# Prévention de l'overfitting, Amélioration de la généralisation...
# shuffle=False : Les données sont passées au modèle dans l'ordre séquentiel. Cela peut être utile pour la validation
# ou les tests, où vous souhaitez des prédictions reproductibles.

# Du coup on va faire le mélange nous même
# Voir shuffle_data() et noter qu'elle modifie le générateur sur place. Elle ne retourne donc rien
# On garde shuffle=False et on invouqe shuffle_data() juste après les créations de img_generator_flow_train et img_generator_flow_validation
# On test avec pred_n_compare2

# On change rien
# On test avec affichage des matrices de confusion
# evaluate_model_1 et evaluate_model_2
# Les matrices de confusion sont identiques (pas top mais identiques)
# Retour dans 04_unfreeze_layers pour appliquer les modifications et voir ce que cela donne


# -----------------------------------------------------------------------------
def explore_type(model, img_generator_flow_validation):

    # 1114 lignes, val = Indice des classes, entre 0 et 3. Ordonnées/Groupées que les 0, puis que les 1...
    y_tmp = img_generator_flow_validation.labels

    # 1114 lignes, val = Indice des classes, entre 0 et 3. Ordonnées/Groupées que les 0, puis que les 1...
    y_true = img_generator_flow_validation.classes

    # 1114 lignes, 4 colonnes, vals = % d'appartenance à la classe
    y_tmp = model.predict(img_generator_flow_validation)

    # 114 lignes, 1 colonne, val = classe
    y_pred = np.argmax(y_tmp, axis=-1)

    # list, 4 classes sous forme de chaines
    class_names = list(img_generator_flow_validation.class_indices.keys())
    return


# -----------------------------------------------------------------------------
def pred_n_compare(model, img_generator_flow_validation):

    # Just to make sure we make prediction with a model no longer in training mode
    model.trainable = False

    # Partie 1
    y_true1 = img_generator_flow_validation.classes
    y_pred1 = np.argmax(model.predict(img_generator_flow_validation), axis=-1)

    # Partie 2
    y_true2 = []
    y_pred2 = []
    nb_image_val = img_generator_flow_validation.n
    batch_size = img_generator_flow_validation.batch_size

    for _ in range(nb_image_val // batch_size):
        val_imgs, val_targets = next(iter(img_generator_flow_validation))
        predictions = model.predict(val_imgs, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)

        if len(val_targets.shape) > 1 and val_targets.shape[1] > 1:
            true_classes = np.argmax(val_targets, axis=1)
        else:
            true_classes = val_targets

        y_pred2 += list(predicted_classes)
        y_true2 += list(true_classes)

    # Sauvegarde des résultats dans un fichier CSV
    df = pd.DataFrame({"y_true1": y_true1, "y_pred1": y_pred1})
    df.to_csv("pred_n_compare1.csv", index=False)

    # On a 114 images et des batchs de 32 images => 34 batchs
    # On aura 1088 preditions 1088 = 32 * 34
    df = pd.DataFrame({"y_true2": y_true2, "y_pred2": y_pred2})
    df.to_csv("pred_n_compare2.csv", index=False)

    # Quand on regarde les résultats dans pred_n_compare1.csv et pred_n_compare2.csv
    # Dans 2, sur 1088 prédictions, 788 sont bonnes 72%
    # Dans 1, sur 1114 prédictions, 307 sont bonnes 27%


# -----------------------------------------------------------------------------
# Mélanger les données de manière cohérente
def shuffle_data(generator):
    indices = np.arange(generator.n)
    np.random.shuffle(indices)
    generator.index_array = indices
    # return generator


# -----------------------------------------------------------------------------
def pred_n_compare2(model, img_generator_flow_validation):

    # Just to make sure we make prediction with a model no longer in training mode
    # model.trainable = False

    # Mélanger les données dans le générateur
    # img_generator_flow_validation = shuffle_data(img_generator_flow_validation)

    # Partie 1
    y_true1 = img_generator_flow_validation.classes
    y_pred1 = np.argmax(model.predict(img_generator_flow_validation), axis=-1)

    # Partie 2
    y_true2 = []
    y_pred2 = []
    nb_image_val = img_generator_flow_validation.n
    batch_size = img_generator_flow_validation.batch_size

    for _ in range(nb_image_val // batch_size):
        val_imgs, val_targets = next(iter(img_generator_flow_validation))
        predictions = model.predict(val_imgs, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)

        if len(val_targets.shape) > 1 and val_targets.shape[1] > 1:
            true_classes = np.argmax(val_targets, axis=1)
        else:
            true_classes = val_targets

        y_pred2 += list(predicted_classes)
        y_true2 += list(true_classes)

    # Sauvegarde des résultats dans un fichier CSV
    df = pd.DataFrame({"y_true1": y_true1, "y_pred1": y_pred1})
    df.to_csv("pred_n_compare1.csv", index=False)

    # On a 114 images et des batchs de 32 images => 34 batchs
    # On aura 1088 preditions 1088 = 32 * 34
    df = pd.DataFrame({"y_true2": y_true2, "y_pred2": y_pred2})
    df.to_csv("pred_n_compare2.csv", index=False)

    # Quand on regarde les résultats dans pred_n_compare1.csv et pred_n_compare2.csv
    # Dans 2, sur 1088 prédictions, sont bonnes  %
    # Dans 1, sur 1114 prédictions,  sont bonnes %


# -----------------------------------------------------------------------------
def evaluate_model_1(model, img_generator_flow_validation):
    # Cette ligne récupère directement les classes vraies de l'ensemble de validation.
    y_true = img_generator_flow_validation.classes

    # Les prédictions sont obtenues en passant tout l'ensemble de validation à la méthode predict du modèle.
    y_pred = np.argmax(model.predict(img_generator_flow_validation), axis=-1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12.0, 10.0))

    class_names = list(img_generator_flow_validation.class_indices.keys())

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    title = f"./img/{timestamp}_confusion_matrix_1.png"
    plt.savefig(title)
    return


# -----------------------------------------------------------------------------
def evaluate_model_2(model, img_generator_flow_validation):

    y_true = []
    y_pred = []
    nb_image_val = img_generator_flow_validation.n

    for _ in range(nb_image_val // k_Batch_Size):
        # Ici, les étiquettes vraies et les prédictions sont récupérées par lots, en itérant sur l'ensemble de validation.
        # Peut causer des incohérences si le générateur d'images ne retourne pas les étiquettes dans le même ordre que img_generator_flow_validation.classes.
        val_imgs, val_targets = next(iter(img_generator_flow_validation))
        # prédit par lots, ce qui peut entraîner des variations si les lots sont traités de manière différente (par exemple, si le générateur n'est pas parfaitement synchronisé).
        predictions = model.predict(val_imgs, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)

        # gestion explicite pour les étiquettes en format multilabel, ce qui peut être crucial si vos étiquettes sont dans ce format.
        if len(val_targets.shape) > 1 and val_targets.shape[1] > 1:
            true_classes = np.argmax(val_targets, axis=1)
        else:
            true_classes = val_targets

        y_pred += list(predicted_classes)
        y_true += list(true_classes)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12.0, 10.0))

    class_names = list(img_generator_flow_validation.class_indices.keys())

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    title = f"./img/{timestamp}_confusion_matrix_2.png"
    plt.savefig(title)
    return


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Setup random generators such that we can compare runs over time
    seed = 0
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Permet de lancer le code depuis un terminal ou de debuger le code dans VSCode
    # Depuis un terminal, je lance VSCode depuis le répertoire ..\skin_project
    # Mais je debug depuis le répertoire ..\skin_project\architecture\02_train_code\03_inceptionV3\05_debug_conf_matrix
    initial_path = os.getcwd()
    prj_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(prj_path)

    os.makedirs("./img", exist_ok=True)

    img_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255, validation_split=0.3
    )

    # On crée 2 flux d'images : entrainement et validation
    train_data_dir = k_data_dir
    img_generator_flow_train = img_generator.flow_from_directory(
        directory=train_data_dir,
        target_size=(k_Img_Height, k_Img_Width),
        batch_size=k_Batch_Size,
        # shuffle=True,
        shuffle=False,
        subset="training",
    )
    # Mélange manuel des données d'entraînement
    shuffle_data(img_generator_flow_train)

    img_generator_flow_validation = img_generator.flow_from_directory(
        directory=train_data_dir,
        target_size=(k_Img_Height, k_Img_Width),
        batch_size=k_Batch_Size,
        # shuffle=True,
        shuffle=False,
        subset="validation",
    )
    # Mélange manuel des données d'entraînement
    shuffle_data(img_generator_flow_validation)

    # num_classes = img_generator_flow_train.num_classes
    # print(f"Number of classes: {num_classes}")

    # Utilisation du modèle pré-entraîné sans fine-tuning
    base_model = tf.keras.applications.InceptionV3(
        input_shape=(k_Img_Height, k_Img_Width, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = tf.keras.Sequential(
        [
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(
                k_nb_classes,
                activation="softmax",
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=k_LearningRate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    model.fit(
        img_generator_flow_train,
        validation_data=img_generator_flow_validation,
        steps_per_epoch=k_steps_per_epoch,
        epochs=k_epochs,
    )

    # explore_type(model, img_generator_flow_validation)
    # pred_n_compare(model, img_generator_flow_validation)

    # pred_n_compare2(model, img_generator_flow_validation)

    evaluate_model_1(model, img_generator_flow_validation)
    evaluate_model_2(model, img_generator_flow_validation)

    os.chdir(initial_path)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# def evaluate_model_0(model, img_generator_flow_validation):

#     # y_true = img_generator_flow_validation.classes
#     # model.predict() retourne un tableau de forme (num_images, num_classes)
#     # num_images est le nombre d'images dans l'ensemble de validation et num_classes est le nombre de classes de classification.
#     # Chaque entrée predictions[i] est donc un vecteur de probabilités pour les différentes classes pour l'image i.
#     # axis=-1 spécifie l'axe le plus interne.
#     # Pour le tableau 2D en sortie de model.predict() cela signifie qu'il trouvera l'indice de la valeur maximale le long de la dernière dimension
#     # => pour chaque vecteur de classes
#     # y_pred = np.argmax(model.predict(img_generator_flow_validation), axis=-1)

#     # Pas besoin de gestion explicite multilabel
#     # labels instead of classes for multilabel

#     print()
#     print(f"From {evaluate_model_0.__name__} :")

#     y_true = img_generator_flow_validation.classes
#     print("y_true :")
#     print(type(y_true))
#     print(y_true.shape)
#     print(y_true)

#     y_pred = model.predict(img_generator_flow_validation)
#     print("y_pred :")
#     print(type(y_pred))
#     print(y_pred.shape)
#     print(y_pred)

#     # Gestion explicite multilabel
#     if len(y_true.shape) > 1 and y_true.shape[1] > 1:
#         y_true = np.argmax(y_true, axis=1)
#         y_pred = np.argmax(y_pred, axis=1)
#     else:
#         y_pred = np.argmax(y_pred, axis=-1)

#     print("y_true :")
#     print(type(y_true))
#     print(y_true.shape)
#     print(y_true)

#     print("y_pred :")
#     print(type(y_pred))
#     print(y_pred.shape)
#     print(y_pred)

#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(12.0, 10.0))

#     class_names = list(img_generator_flow_validation.class_indices.keys())

#     sns.heatmap(
#         cm,
#         annot=True,
#         fmt="d",
#         cmap="Blues",
#         cbar=False,
#         xticklabels=class_names,
#         yticklabels=class_names,
#     )
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     title = f"./img/{timestamp}_confusion_matrix_0.png"
#     plt.savefig(title)
#     return


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# def evaluate_model_1(model, img_generator_flow_validation):

# y_true = img_generator_flow_validation.classes
# model.predict() retourne un tableau de forme (num_images, num_classes)
# num_images est le nombre d'images dans l'ensemble de validation et num_classes est le nombre de classes de classification.
# Chaque entrée predictions[i] est donc un vecteur de probabilités pour les différentes classes pour l'image i.
# axis=-1 spécifie l'axe le plus interne.
# Pour le tableau 2D en sortie de model.predict() cela signifie qu'il trouvera l'indice de la valeur maximale le long de la dernière dimension
# => pour chaque vecteur de classes
# y_pred = np.argmax(model.predict(img_generator_flow_validation), axis=-1)

# Pas besoin de gestion explicite multilabel
# labels instead of classes for multilabel
# print()
# print(f"From {evaluate_model_1.__name__} :")

# y_true = img_generator_flow_validation.labels
# print("y_true :")
# print(type(y_true))
# print(y_true.shape)
# print(y_true)

# y_pred = model.predict(img_generator_flow_validation)
# print("y_pred :")
# print(type(y_pred))
# print(y_pred.shape)
# print(y_pred)

# if len(y_true.shape) > 1 and y_true.shape[1] > 1:
#     y_true = np.argmax(y_true, axis=1)
#     y_pred = np.argmax(y_pred, axis=1)
# else:
#     y_pred = np.argmax(y_pred, axis=-1)

# y_true = img_generator_flow_validation.labels
# print("y_true :")
# print(type(y_true))
# print(y_true.shape)
# print(y_true)

# y_pred = model.predict(img_generator_flow_validation)
# print("y_pred :")
# print(type(y_pred))
# print(y_pred.shape)
# print(y_pred)

# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(12.0, 10.0))

# class_names = list(img_generator_flow_validation.class_indices.keys())

# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="Blues",
#     cbar=False,
#     xticklabels=class_names,
#     yticklabels=class_names,
# )
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# title = f"./img/{timestamp}_confusion_matrix_1.png"
# plt.savefig(title)
# return


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# def evaluate_model_2(model, img_generator_flow_validation):

#     y_true = []
#     y_pred = []

#     # logger.debug(f"Nb images valiadation = {img_generator_flow_validation.n}")
#     # nb_image_val = 1114
#     nb_image_val = img_generator_flow_validation.n
#     batch_size = img_generator_flow_validation.batch_size

#     # Gestion explicite multi-label
#     for _ in range(nb_image_val // batch_size):
#         val_imgs, val_targets = next(iter(img_generator_flow_validation))
#         predictions = model.predict(val_imgs, verbose=0)
#         predicted_classes = np.argmax(predictions, axis=1)

#         # Convert val_targets to class indices if it's in multilabel-indicator format
#         # Confirmé : on est en multi label
#         if len(val_targets.shape) > 1 and val_targets.shape[1] > 1:
#             true_classes = np.argmax(val_targets, axis=1)
#         else:
#             true_classes = val_targets

#         y_pred += list(predicted_classes)
#         y_true += list(true_classes)

#     print()
#     print(f"From {evaluate_model_2.__name__} :")

#     # y_true = img_generator_flow_validation.labels
#     # y_pred = model.predict(img_generator_flow_validation)

#     # Sans gestion explicite multilabel => MARCHE PAS
#     # for _ in range(nb_image_val // batch_size):
#     #     val_imgs, val_targets = next(iter(img_generator_flow_validation))
#     #     predictions = model.predict(val_imgs, verbose=0)
#     #     predicted_classes = np.argmax(predictions, axis=-1)

#     #     # Directly append the true classes and predicted classes
#     #     y_true += list(val_targets)
#     #     y_pred += list(predicted_classes)

#     print("y_true :")
#     print(type(y_true))
#     print(y_true.len())
#     print(y_true)

#     print("y_pred :")
#     print(type(y_pred))
#     print(y_pred.len())
#     print(y_pred)

#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(12.0, 10.0))

#     class_names = list(img_generator_flow_validation.class_indices.keys())
#     sns.heatmap(
#         cm,
#         annot=True,
#         fmt="d",
#         cmap="Blues",
#         cbar=False,
#         xticklabels=class_names,
#         yticklabels=class_names,
#     )
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     title = f"./img/{timestamp}_confusion_matrix_2.png"
#     plt.savefig(title)
#     return
