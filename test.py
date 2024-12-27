import requests

url = "http://127.0.0.1:8000/predict/"
file_path = "/home/wsluser/Mywork/Portfolio/Plant-seedling-classification-project/0021e90e4.png"

# Open the image file in binary mode
with open(file_path, "rb") as image_file:
    # Send the file in a POST request
    response = requests.post(url, files={"file": image_file})

# Print the response from the server
print(response.json())