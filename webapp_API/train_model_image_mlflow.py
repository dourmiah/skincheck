import requests
import os

mlflow_server_url = 'https://your-heroku-app.herokuapp.com'
model_path = 'D:/Quentin/jedha/jedhaFullStack/skin_project/webapp_API/my_app/src/model/cat_classifier.h5'
model_name = 'cat_classifier'
run_id = 'f5f6ea3fe18f483197a46f8fb00c40a7'

headers = {
    'Content-Type': 'application/json',
}

# Créer une nouvelle run
create_run_url = f'{mlflow_server_url}/api/2.0/mlflow/runs/create'
response = requests.post(create_run_url, headers=headers)
response_data = response.json()
new_run_id = response_data['run']['info']['run_id']

# Uploader le modèle
log_model_url = f'{mlflow_server_url}/api/2.0/mlflow/runs/log-model'
files = {'file': open(model_path, 'rb')}
data = {
    'run_id': new_run_id,
    'artifact_path': model_name,
}
response = requests.post(log_model_url, headers=headers, files=files, data=data)

if response.status_code == 200:
    print('Model uploaded successfully!')
else:
    print(f'Failed to upload model: {response.content}')
