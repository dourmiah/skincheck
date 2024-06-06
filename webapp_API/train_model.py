import os
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Charger les données
data = load_iris()
X, y = data.data, data.target

# Entraîner le modèle
model = RandomForestClassifier()
model.fit(X, y)

# Spécifiez le chemin absolu où sauvegarder le modèle
absolute_model_path = 'G:/Mon Drive/projet/skin_project/webapp_API/my_app/src/model/model.pkl'

# Assurez-vous que le répertoire existe
os.makedirs(os.path.dirname(absolute_model_path), exist_ok=True)

# Sauvegarder le modèle
joblib.dump(model, absolute_model_path)
print(f"Model saved to {absolute_model_path}")
