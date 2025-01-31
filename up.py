import csv
import os
import re
import cv2
import time
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from scipy.spatial.distance import cosine

app = Flask(__name__)

# Folder where the university logos are stored
LOGO_FOLDER = r'C:\Users\inc3061\Documents\Renamed_logo'

# Path to the CSV file containing university names
CSV_FILE_PATH = 'static/universities.csv'

# Ensure the output folder for detected logos exists
DETECTED_OBJECT_FOLDER = os.path.join('static', 'detected_object')
os.makedirs(DETECTED_OBJECT_FOLDER, exist_ok=True)

# Ensure the folder for uploaded images exists
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Folder to save the processed logos
SAVED_LOGOS_FOLDER = os.path.join('static', 'saved_logos')
os.makedirs(SAVED_LOGOS_FOLDER, exist_ok=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to read universities from the CSV file
def read_universities_from_csv(csv_file_path):
    universities = []
    with open(csv_file_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            universities.append(row['university_name'])
    return universities

# Function to clean and normalize file names (remove numbers and special characters)
def normalize_name(name):
    return re.sub(r'[^a-zA-Z\s]', '', name).lower().replace(" ", "")

# Function to find logos for each university
def find_logos_for_universities(university_name, logo_folder):
    normalized_university_name = normalize_name(university_name)
    logo_file_details = []
    
    for file in os.listdir(logo_folder):
        normalized_logo_filename = normalize_name(file)
        
        if normalized_university_name in normalized_logo_filename and file.lower().endswith(('png', 'jpg', 'jpeg', 'gif')):
            logo_file_path = os.path.join(logo_folder, file)
            
            # Save the file to the SAVED_LOGOS_FOLDER
            saved_logo_path = os.path.join(SAVED_LOGOS_FOLDER, file)
            if not os.path.exists(saved_logo_path):  # To avoid overwriting existing logos
                # Copy the logo to the saved folder
                with open(logo_file_path, 'rb') as fsrc, open(saved_logo_path, 'wb') as fdst:
                    fdst.write(fsrc.read())
            
            logo_file_details.append({'filename': file, 'file_path': saved_logo_path})
    
    return logo_file_details

# Function to check if a file is an allowed image type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to process the image in a specific region
def process_region(image, x, y, w, h, padding_top=50, padding_bottom=50):
    roi = image[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    _, roi_thresh = cv2.threshold(roi_gray, 200, 255, cv2.THRESH_BINARY_INV)
    roi_contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in roi_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 100 < w < 400 and 100 < h < 400:
            y2 = min(y + h + padding_bottom, roi.shape[0])
            return roi[y:y2, x:x+w]
    return None

# Function to resize the extracted logo
def resize_image(image, target_width, target_height):
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

# Function to calculate SSIM
def calculate_ssim(imageA, imageB):
    # Convert images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    
    # Resize imageB to match the dimensions of imageA
    grayB_resized = cv2.resize(grayB, (grayA.shape[1], grayA.shape[0]))
    
    # Compute SSIM between the two images
    score, _ = ssim(grayA, grayB_resized, full=True)
    return score

# Function to extract features using VGG16 and calculate cosine similarity
def calculate_feature_similarity(image_path1, image_path2):
    # Load pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
    
    # Function to preprocess image for VGG16
    def preprocess_image(image_path):
        img = image.load_img(image_path, target_size=(224, 224))  # Resize to 224x224 for VGG16
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        return img_data
    
    # Preprocess the two images
    img1 = preprocess_image(image_path1)
    img2 = preprocess_image(image_path2)
    
    # Extract features
    features_img1 = model.predict(img1)
    features_img2 = model.predict(img2)
    
    # Flatten features and calculate cosine similarity
    features_img1 = features_img1.flatten()
    features_img2 = features_img2.flatten()
    similarity = 1 - cosine(features_img1, features_img2)
    
    return similarity

# Route to handle image upload and processing
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'logo' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['logo']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type, only png, jpg, jpeg, and gif are allowed'})
    
    # Save the uploaded file in the uploads folder
    filename = secure_filename(file.filename)
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)
    
    # Process the image
    start_time = time.time()
    
    image = cv2.imread(upload_path)
    
    if image is None:
        return jsonify({'error': f"Image '{filename}' could not be loaded. Skipping."})
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    height, width = image.shape[:2]
    regions = {
        'top': (5, 0, width, height // 5),
        'bottom': (0, height - height // 5, width, height),
        'left': (0, 0, width // 5, height),
        'right': (width - width // 5, 0, width, height),
        'center': (width // 4, height // 4, width // 2, height // 2)
    }
    
    detected_in_any_region = False
    for region_name, (x, y, w, h) in regions.items():
        roi_logo = process_region(image, x, y, w, h, padding_bottom=70)
        if roi_logo is not None:
            detected_in_any_region = True
            resized_logo = resize_image(roi_logo, 500, 500)
            
            # Save the detected logo in the detected_object folder
            output_path = os.path.join(DETECTED_OBJECT_FOLDER, f"{os.path.splitext(filename)[0]}_logo_{region_name}.jpg")
            cv2.imwrite(output_path, resized_logo)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    if detected_in_any_region:
        # Call the function to compare and calculate similarities after the logo is saved
        compare_and_calculate_similarity()

        return jsonify({'message': f"Logo detected and saved.", 'execution_time': f"{execution_time:.4f} seconds"})
    else:
        return jsonify({'message': 'No logo detected in the specified regions for this image.', 'execution_time': f"{execution_time:.4f} seconds"})

# Route to display the homepage
@app.route('/')
def index():
    universities = read_universities_from_csv(CSV_FILE_PATH)
    return render_template('index2.html', universities=universities)

# Route to handle getting logos based on selected university
@app.route('/get_logos', methods=['POST'])
def get_logos():
    university_name = request.form.get('university_name')
    logo_details = find_logos_for_universities(university_name, LOGO_FOLDER)
    return jsonify(logo_details)

# Compare detected logos with saved logos and calculate similarity
def compare_and_calculate_similarity():
    # Get all image files in the detected_object folder
    detected_files = [f for f in os.listdir(DETECTED_OBJECT_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    saved_files = [f for f in os.listdir(SAVED_LOGOS_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    results = []

    for detected_file in detected_files:
        detected_path = os.path.join(DETECTED_OBJECT_FOLDER, detected_file)

        for saved_file in saved_files:
            saved_path = os.path.join(SAVED_LOGOS_FOLDER, saved_file)
            
            # Calculate feature similarity using VGG16
            feature_similarity = calculate_feature_similarity(detected_path, saved_path)
            
            # Calculate SSIM
            detected_image = cv2.imread(detected_path)
            saved_image = cv2.imread(saved_path)
            ssim_score = calculate_ssim(detected_image, saved_image)
            
            # Store the result
            results.append([detected_file, saved_file, feature_similarity, ssim_score])
    
    # Save the results to a CSV file
    output_csv_path = os.path.join('static', 'Cosine_score.csv')
    df = pd.DataFrame(results, columns=["Detected Filename", "Saved Filename", "Cosine Similarity", "SSIM Score"])
    df.to_csv(output_csv_path, index=False)

    print(f"Results have been saved to '{output_csv_path}'.")

if __name__ == '__main__':
    app.run(debug=True)
