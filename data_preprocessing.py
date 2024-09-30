import os
import cv2
import numpy as np
from datetime import datetime

# Set dataset directory (use raw string to handle Windows-style paths)
dataset_dir = r"D:\Downloads\Age Recognition\dataset"
img_size = 128  # Resize images

# Function to calculate age from a birthdate
def calculate_age(birthdate):
    try:
        birthdate = datetime.strptime(birthdate, "%Y-%m-%d")
        today = datetime.today()
        return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    except ValueError:
        # Skip invalid birthdate formats
        return None

# Function to read and preprocess the images
def load_data():
    images = []
    ages = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(dataset_dir, filename))
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            # Assuming filename contains birthdate, e.g., 'name_1953-10-00.jpg'
            birthdate = filename.split("_")[1].split(".")[0]  # Extract birthdate
            age = calculate_age(birthdate)  # Calculate age from birthdate
            if age is not None:  # Skip if age couldn't be calculated
                ages.append(age)
    return np.array(images), np.array(ages)

# Call the function to load data
X, y = load_data()

# Check if data was loaded successfully
if len(X) > 0 and len(y) > 0:
    # Save data for training (optional)
    np.save("X.npy", X)
    np.save("y.npy", y)
    print("Data preprocessing complete and saved.")
else:
    print("No valid data found.")
