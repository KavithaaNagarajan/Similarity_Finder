# Similarity_Finder
![image](https://github.com/user-attachments/assets/ff3a9795-bbc5-485d-b673-7ce8b4010a56)

![image](https://github.com/user-attachments/assets/68981f15-3b20-4a0a-81ea-269d407e95fb)

![image](https://github.com/user-attachments/assets/7267cf81-f489-43fd-8651-1534c131838c)
# University Logo Detection and Similarity Comparison
## This project is a web application that allows users to upload an image, detect logos in the image, and compare those logos with a predefined set of university logos. The logos are compared using two methods:

Cosine Similarity - Extracts features using a pre-trained VGG16 model.
Structural Similarity Index (SSIM) - Measures image similarity by comparing pixel-level structure.
# Features
Logo Detection: The app processes uploaded images and identifies potential logos based on predefined regions (e.g., top, bottom, center) within the image.
Logo Comparison: After detecting a logo in an image, the app compares it with logos from a local database of university logos, calculating the similarity using both cosine similarity (based on features extracted via VGG16) and SSIM.
CSV Report: The app generates a CSV report with the comparison results for each logo detected in the uploaded image, showing cosine similarity and SSIM scores for each pair.
# Requirements
### 1.Python 3.x
### 2.Flask (Web framework)
### 3.OpenCV (Computer vision library)
### 4.TensorFlow (Deep learning framework)
### 5.scikit-image (For SSIM calculation)
### 6.pandas (For CSV file handling)
You can install the required dependencies by running:

### bash
### Copy
### pip install -r requirements.txt
# Folder Structure
### The application relies on several folders to store images and results:

### /static/uploads/: Folder where uploaded images are temporarily stored.
### /static/detected_object/: Folder where detected logos from images are saved.
### /static/saved_logos/: Folder containing the university logos to compare against.
### /static/Cosine_score.csv: The output file that stores comparison results, including cosine similarity and SSIM scores.
# How It Works
### 1. Upload an Image
The user can upload an image containing a logo.
The app processes the image and searches for potential logos in predefined regions such as the top, bottom, left, right, and center.
Detected logos are saved into the detected_object folder.
### 2. Logo Detection
The uploaded image is converted to grayscale and blurred.
A binary threshold is applied to detect contours in the image.
The regions of the image where logos might be present are specified, and the image within these regions is analyzed to identify the logo.
### 3. Logo Comparison
For each detected logo, the app compares it with logos stored in the saved_logos folder.
Two methods are used to calculate similarity:
Cosine Similarity using VGG16 features: The app extracts features from both the detected and saved logos using the pre-trained VGG16 model and calculates cosine similarity between them.
SSIM (Structural Similarity Index): Compares the structural similarity between the detected and saved logos at the pixel level.
The similarity scores are stored in a CSV file (Cosine_score.csv).
### 4. CSV Report
The results of the similarity comparison are saved to a CSV file, which includes:
Detected Filename: The name of the detected logo file.
Saved Filename: The name of the saved logo file from the university database.
Cosine Similarity: The cosine similarity score between the detected and saved logo.
SSIM Score: The SSIM score between the detected and saved logo.
# Routes
### 1. Home (/)
Displays the main page where users can select a university from a list (loaded from a CSV file).
### 2. Upload Image (/upload)
Handles the image upload and processing.
If logos are detected in the uploaded image, they are saved and compared with the saved logos.
### 3. Get Logos (/get_logos)
Given a university name, the app returns a list of logos stored in the saved_logos folder for that university.
# How to Run the Application
## Start the Flask server:

To run the application, navigate to the directory where the Python script is located and run:
### bash
### Copy
### python app.py
This will start the Flask development server, which you can access at http://localhost:5000.
### Upload an Image:

Go to the homepage (/), select a university from the list, or upload your own image containing a logo.
The app will process the image, detect logos, and compare them with the saved logos.
### Download Comparison Results:

After processing the image, you can download the comparison results from Cosine_score.csv.
### Known Issues
The app may not detect logos accurately if the uploaded image is of poor quality or contains complex backgrounds.
The logo database may need to be updated with more logos for better comparison results.
### Credits
This project uses VGG16 from TensorFlow for feature extraction and OpenCV for image processing.
scikit-image is used to calculate SSIM for image similarity measurement.
