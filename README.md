# Plant-seedling-classifcation

This work concerns image processing. I use multiple image processing techniques to segment and classify plant seed. I detail my approach here : "https://medium.com/@majdii.karim/plant-seedling-classification-project-ec22d59d09d2"

Following are the steps to run the project on a local machine :

- clone the repo.

- create conda environement, I'm used python 3.12.8.

- pip install -r requirements.txt

- conda activate env

- You shou start the localhost server on a choosen port (8000 for instance). The API will now be ready to receive and process image classification requests :

uvicorn main:app --reload --host 0.0.0.0 --port 8000

- send a post request to the local server. 

curl -X POST "http://127.0.0.1:8000/predict/" -H "accept: application/json" -H "Content-Type: multipart/form-data" 
-F "file=@pathtoimage"

You must keep the @ in order to use cURL instead of the regular str path. This will allow to format the input into an instance object of the class File in the package Fastapi that specializes in aqcuiring files as inputs to the API. "@" should be followed by the absolute path.

- Use the test.py script to test the API by sending a post request to your localhost server connected to the port 8000.

python test.py
