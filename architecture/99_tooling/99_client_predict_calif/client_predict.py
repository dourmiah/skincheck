import pandas as pd
import mlflow
import boto3
import os

from botocore.exceptions import NoCredentialsError

try:
    boto3.setup_default_session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )
except NoCredentialsError:
    print("Please make sure to run ./secrets.ps1 before to run this script.")

mlflow.set_tracking_uri("https://skincheck-tracking-server-6e98556bcc6b.herokuapp.com/")

logged_model = "runs:/1bd65c9fc7d7412fa66efe47cea271c1/modeling_housing_market"

loaded_model = mlflow.pyfunc.load_model(logged_model)

# Content of the csv file
# MedInc	HouseAge	AveRooms	AveBedrms	Population	AveOccup	Latitude	Longitude	MedHouseVal
# 8.3252	41.0	    6.98    	1.02        322.0	    2.55	    37.88	    -122.23	    4.52
# 8.3014	21.0	    6.23    	0.97        2401.0	    2.10	    37.86	    -122.22	    3.58
# 7.2574	52.0	    8.28    	1.07        496.0	    2.80	    37.85   	-122.24	    3.52
# 5.6431	52.0	    5.81    	1.07        558.0	    2.54	    37.85	    -122.25	    3.41

data = {
    "MedInc": [8.32],
    "HouseAge": [41.0],
    "AveRooms": [6.98],
    "AveBedrms": [1.02],
    "Population": [322.0],
    "AveOccup": [2.55],
    "Latitude": [37.88],
    "Longitude": [-122.23],
}

input_df = pd.DataFrame(data)
predictions = loaded_model.predict(input_df)
print("Prédictions : ", predictions)
