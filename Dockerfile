
# Start from a base image
FROM python:3.8

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the application code into the container
COPY ["house_prediction_model-no_drop_features.pkl", "app_house_1.py", "./"] .

# Expose the app port
EXPOSE 8000

# Run command
CMD ["uvicorn", "app_house_1:app", "--host", "0.0.0.0"]
