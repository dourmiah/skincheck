# MLFlow Tracking

Répertoire sincheck-mlflow-tracking-server
créer Dockerfile
créer requirements.txt

Commande
docker build -t skincheck-tracking-server .


Commandes
heroku login
heroku create skincheck-tracking-server 
heroku container:login 
heroku container:push web -a skincheck-tracking-server 
heroku container:release web -a skincheck-tracking-server 
heroku open -a skincheck-tracking-server 


AWS
groupe  : skincheck-group
user    : skincheck-user


S3
bucket : skincheck-bucket (ACLs enabled - Access Control List)

répertoires :
    skincheck-dataset
    skincheck-artifacts copier l'URI S3




# SinCheck Trainer
Répertoire sincheck-trainer
créer Dockerfile
créer requirements.txt

Commande
docker build -t skincheck-trainer .





* conda install mlflow
* conda update -c conda-forge paramiko
* conda install requests=2.31.0
    * https://stackoverflow.com/questions/64952238/docker-errors-dockerexception-error-while-fetching-server-api-version






Rendre un répertoire accessible en lecture seule sur S3

```
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::skincheck-bucket/skincheck-dataset/*"
    }
  ]
}
```

Faut se rappeler que AWS_ACCESS_KEY_ID et AWS_SECRET_ACCESS_KEY
Sont spécifiques à un répertoire du bucket sur S3
Tout se passe comme si MLFlow n'avait besoin que de ces 2 informations pour aller taper le répertoire en question

Par contre sur Heroku le serveur MLFlow Tracking à besoin des 2 plus de l'URI S3 : s3://skincheck-bucket/skincheck-artifacts/
Je suis pas sur de comprendre



Var d'environnement MLFLOW
* https://mlflow.org/docs/latest/python_api/mlflow.environment_variables.html