# Use the official Python image as the base image
FROM python:3.11-slim

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Set the command to run your application using Gunicorn
CMD ["gunicorn", "--timeout", "1000000", "--bind", "0.0.0.0:8080", "app:app"]