# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

ENV PYTHONPATH="/"

# Specify the command to run your application
# Replace 'app.py' with the entry point of your application
CMD ["python", "explainability_app.py"]

# Expose a port if your application requires it (optional)
# EXPOSE 5000