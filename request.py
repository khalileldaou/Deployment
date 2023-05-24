import json
import requests

# Here are the sentence vectors which will be passed to the function
data = {"sentence": "This is a trial"}

# The sentence vectors included in json format
json_data = json.dumps(data)
# The headers are required for the post request
headers = {"Content-Type": "application/json"}
# The address of the local server we are posting to
local_server = "http://localhost:8080/predict"


# Defining a function to post the data to the local server and receive a response
def send_request(json_data):
    response = requests.post(local_server, data=json_data, headers=headers)
    output = response.text
    return output


print(send_request(json_data))
