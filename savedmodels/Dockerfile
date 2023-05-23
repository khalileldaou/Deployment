# Use an official Python runtime as the base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app/savedmodels

# Copy the requirements file to the working directory
COPY requirements.txt /app/savedmodels

# Install the required packages
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app/savedmodel
COPY . /app/savedmodels

# Set the command to run demo.py
CMD ["python", "demo.py"]