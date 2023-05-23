# Use an official Python runtime as the base image
FROM python:3.9-slim-buster AS savedmodels

# Copy the savedmodels directory to the container
COPY savedmodels /app/savedmodels


# Use the savedmodels image as the base for the Flask application
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the required packages
RUN pip install -r requirements.txt

# Copy the main.py file to the container
COPY . .

# Copy the templates directory to the container
COPY templates /app/templates

# Copy the savedmodels directory from the savedmodels image
COPY --from=savedmodels /app/savedmodels /app/savedmodels

# Expose port 8080 for the Flask app
EXPOSE 8080

# Set the entrypoint command to run the Flask app
CMD ["python", "main.py"]