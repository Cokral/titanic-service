# titanic-service

API to serve a titanic classification model.

This was made as a technical test for dataprovider.com.

## How to use 

To use, just build the docker-compose using the following command:
```
docker-compose up --build
```
It will build the project and run the API.

You can then go to the adress:
```
http://localhost:12000/docs#/
```
to visit and use the API.

## What can I do

 - `POST /train`
Trains and saves a classifier. You can specify the model name and the parameters.
 - `GET /models`
Returns the models already trained and their metadata (name, training metrics and parameters).
 - `POST /predict`
Returns the prediction of the specified model for the parameters.



