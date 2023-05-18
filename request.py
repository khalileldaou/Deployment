import os
import json
from flask import Flask, Response, request, render_template, session

# Here are the sentence vectors which will be passed to the function
data = {'sentence': "This is a trial"}

# The sentence vectors included in json format
data_json = json.dumps(data)
# The headers are required for the post request
headers = {"Content-Type": "application/json"}
# The address of the local server we are posting to
local_server = 'http://localhost8080/predict'

# Defining a function to post the data to the local server and receive a response
def send_request(json_input):
    response = request.post(local_server, data=json_input, headers=headers)
    output = json.loads(response.text)
    return output

