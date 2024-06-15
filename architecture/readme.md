There are 3 main directories :
1. mlflow_tracking_server :
    * To build and deploy mlflow tracking server on Heroku
1. images for models trainers
    * most of the model training code run into docker images
    * there are 3 images in 3 differents directories because the training code can be based on sklearn, tensorflow, tensorflow with GPU support
    * Each of them requires a specifi configuration described mostly in the requirements.txt file
    * tensorflow_trainer_v2 add pydot and graphiz support to tensorflow_trainer image. 
        * Used to draw and understand how layers are organized in inception and determinse which layer to unfreeze
        * see architecture\02_train_code\03_inceptionV3\04_unfreeze_layers
    * each directory include a "build_blablabla.ps1" which should be invoked from a terminal in the current  directory
1. train_code
    * there are 3 directories each hosting training code based on skelearn, tensorflow or reusing inceptionV3 (tensorflow)
    * templates directory (the 4th directory) contains step by step instruction on how to create your own training code
    * Each of the 3 first directories contains a run_experiment.ps1 that should be invoked for the current directory in a terminal
    * Read the script, but basically :
        * it first invoque secrets.ps1 in order to set the required environment variables
        * you may want to update the content of secrets.ps1 with the name of your project for example 
        * otherwise don't touch 
        * **IMPORTANT** never share or expose ``secrets.ps1`` (double check the content of you .gitignore **BEFORE** any commit/sync)
        * the script then calls mlflow run ... with or without parameters you may want to pass to train.py 
        * train.py is the code in charge of the training of the model
        * it is executed in the context of the Docker image lested in the MLproject file (see ``image:`` entry)
        * the Docker image must be available.
            * if needed, build it using scripts available in `architecture\01_images_for_model_trainers`
1. tooling
    * tools and test code developped during the project
1. data_4
    * Dataset with 4 classes
    * Helps to speed up some testing
    * Should be a copy of the first 3 classes of data_24, plus healthy_skins directory
1. data_24
    * 24 classes
    * no spaces in names, the word "photos" is removed, " " replace with "_"
    * see architecture\99_tooling\01_rename_classes
    * some names are modified "by hand"
