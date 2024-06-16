### Last update : June 16 2024

There are 3 main directories :

1. ##  mlflow_tracking_server directory : 
    * To build and deploy mlflow tracking server on Heroku
1. ##  images for models trainers directory : 
    * most of the model training code run into docker images
    * there are 3 images in 3 differents directories because the training code can be based on sklearn, tensorflow, tensorflow with GPU support
    * yes, there is a 4th directory named `tensorflow trainer v2`. At the time of writing it is a specific image which includes tools that help to display the inner content of InceptionV3. Forget this directory.
    * Each of the image requires a specific configuration described mostly in the `requirements.txt` file which is available in each directory.
    * For example, `tensorflow_trainer_v2` add `pydot` and `graphiz` support to tensorflow_trainer image. 
        * Used to draw and understand how layers are organized in inceptionV3 and determinse which layer to unfreeze
        * see `architecture\02_train_code\03_inceptionV3\04_unfreeze_layers`
    * each directory include a "build_blablabla.ps1" which should be invoked from a terminal in the current  directory
    * My recommendation : if you need to modify some parameters of the command line etc. Modify the `build_xxx.ps1` script. On the long run it is more effective to use a script rather than to remember which specific options goes with which configuration.
1. ## train_code directory : 
    * there are 3 directories, each of them hosting training code based on skelearn, tensorflow or reusing inceptionV3 (tensorflow)
    * templates directory (the 4th directory) contains step by step instructions on how to create your own training code
    * Each of the 3 first directories contains a run_experiment.ps1 that should be invoked for the current directory in a terminal
    * Read the script, but basically :
        * it first invoque secrets.ps1 in order to set the required environment variables
        * you may want to update the content of secrets.ps1 with the name of your project for example 
        * otherwise don't touch ``secrets.ps1`` 
        * **VERY IMPORTANT** never share nor expose ``secrets.ps1`` (double check the content of your .gitignore **BEFORE** any commit/sync)
        * the script then calls mlflow run ... with or without parameters that you may want to pass to the train.py script
        * train.py is the code in charge of the training of the model
        * it is executed in the context of the Docker image listed in the MLproject file (see ``image:`` entry)
        * Obviously, the Docker image in which the `train.py` will be executed must be available before you call `run_experiment.ps1`
            * if needed, build it using scripts available in `architecture\01_images_for_model_trainers`
1. ## tooling directory : 
    * tools and test code developped during the project
1. ## data_4 directory : 
    * Dataset with only 4 classes
    * This directory should be added to your `.gitignore`
    * Helps to speed up some testing
    * Should be a copy of the first 3 classes of data_24, plus healthy_skins directory
1. ## data_24 directory : 
    * 24 classes
    * This directory should be added to your `.gitignore`
    * no spaces in names, the word "photos" is removed, " " replace with "_"
    * see architecture\99_tooling\01_rename_classes
    * some names are modified "by hand"
