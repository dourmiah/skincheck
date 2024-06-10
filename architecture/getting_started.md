<!-- 
1. Ouvrir un terminal
1. Passer dans l'environnement  
-->

# Créer une image où on executera le code de training
1. On est sous VSCode
1. Aller dans le répertoire : architecture\skincheck_trainer_TF
1. Y ouvrir une console
1. Saisir la commande : ./build_skincheck_trainer_tf.ps1
1. A la fin, pour vérifier que l'image est disponible,  saisir la commande : docker image ls


# Tester que le template de code s'execute bien
1. Aller dans le répertoire : architecture\TF_local_2
Il y a 4 fichiers
* MLproject : va utiliser l'image skincheck_trainer_tf pour lancer le script d'entrainement (train.py)
    * On ne touche à rien pour l'instant mais plus tard on pourra y customiser :    
        * name: voir le "californian_housing_market" en haut du fichier
        * les paramètres : voir "epochs" et "batch_size"
* run_experiment.ps1
    * c'est le script qu'on va lancer
    * Il s'assure que les variables d'environnement sont bien définies et de lancer mlflow qui va construire l'image et lancer le script dans l'image
* secrets.ps1
    * PAS TOUCHE
    * Il est très important que secrets.ps1 soit dans le .gitignore
* train.py
    * C'est le code qui 
        * Va chercher si besoin les données sur le busket S3 
            * Attention à ne pas abuser. C'est philippe qui paie. 
        * Entraine le modele
        * Chrnometre chaque étape
        * Sauve les artifacts sur S3 ainsi que les paramatres et les chronos sur le serveur mlflow tracking
1. Ouvrir une console
1. Saisir la commande : ./run_experiment.ps1
1. A la fin aller sur le serveur mlflow tracking et y retrouver les résultats du test
    * https://skincheck-tracking-server-6e98556bcc6b.herokuapp.com/


# Modifier le template pour entrainer votre modèle, enregistrer ses artifacts et ses paramètres
1. Copier coller le répertoire : architecture\TF_local_2
1. Ourir MLproject
    * Adapter le "name", mettre "skincheck" (plutôt que californian_housing_market) 
    * Modifier peut être les valeurs et/ou quantitites des paramètres
1. Ouvrir train.py et modifier le code
    * Voir section : Comment modifier le code du template
1. Saisir la commande : ./run_experiment.ps1
1. A la fin aller sur le serveur mlflow tracking et y retrouver les résultats du test
    * https://skincheck-tracking-server-6e98556bcc6b.herokuapp.com/


Si plus tard vous souhaitez passez des paramètres avec des valeurs autres que celles par défaut
1. Ouvrir run_experiments
1. Commenter/decommenter la ligne qui va bien
1. Inspirez vous de la ligne pour passer vos paramètres complémentaires
1. Changer les noms paramètres et les valeurs


# Comment modifier le code du template?
1. C'est un code objet très scolaire/didactique
1. Tout en haut, il faut adapter la constante 
    * k_RelativePath = "modeling_housing_market"
    * C'est sous ce nom que seront regroupés tous vos runs

1. Dans la classe ModelTrainer

    Dans la plupart de méthode, le code à modifier se trouve entre les 2 lignes
    ```
    start_time = time.time()
    ...
    mlflow.log_metric("load_data_time", time.time() - start_time)
    ```
    Il est préférable de ne pas toucher ces lignes et de faire en sorte que les valeurs retournées soient les mêmes que celles du template.

    1. Méthode `__init__`
        * sauver les paramètres supplémenaires si il y en a
        * aller voir dans main comment on instancie un objet de classe ModelTrainer avant d'invoquer .run()
    
    1. Méthode `load_data()`
        * Vous chargez vos données dans data

    1. Méthode `preprocess_data()`
        * C'est là que vous traitez les données

    1. Méthode `build_model()`
        * Vous précisez l'organisation de votre modèle

    1. Méthode `train_model()`
        * Normalement vous devriez avoir rien à faire
        
    1. Méthode `evaluate_model()`
        * Normalement vous devriez avoir rien à faire

    1. Méthode `log_parameters()`
        * Normalement vous devriez avoir rien à faire
        * Si vous avez des paramètres supplémentaires ajoutez les

    1. Méthode `log_model()`
        * Normalement vous devriez avoir rien à faire

    1. Méthode `run()`
        * Elle appelle les autres méthodes en passant les paramètres
        * Normalement vous devriez avoir rien à faire

1. la fonction `__main__`
    * Normalement vous devriez avoir rien à faire
        * Si vous avez gardé epochs et batch_size en paramètre 
    * Sinon fua ts'inspirer du code existant pour inclure vos propres paramètres
