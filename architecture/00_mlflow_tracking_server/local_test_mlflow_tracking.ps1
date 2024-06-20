# C'est pour faire un test en local

# Pour faire le test en local, bin sûr il faut que l'image skincheck-tracking-server soit disponible
#       docker build -t skincheck_tracking_server .

# Pas besoin de préciser AWS_ACCESS_KEY_ID etc.

docker run -it `
-p 4000:4000 `
-v "$(pwd):/home/app" `
-e APP_URI=APP_URI `
-e PORT=4000 `
-e AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID `
-e AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY `
skincheck_mlflow_tracking_server