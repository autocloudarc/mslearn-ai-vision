# ===== IMPORTS =====
# Azure Custom Vision SDK for training object detection models
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
# Models for creating image batches with tagged regions (for object detection)
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
# Authentication module for API key-based credentials
from msrest.authentication import ApiKeyCredentials
import time  # For handling delays if needed
import json  # For parsing the tagged-images.json file
import os   # For environment variables and file operations

def main():
    """
    Main entry point for uploading tagged images to Azure Custom Vision for object detection.
    
    This script:
    1. Loads Azure credentials and project settings from environment variables (.env file)
    2. Authenticates with the Azure Custom Vision training service
    3. Retrieves the Custom Vision project configuration
    4. Uploads images with object detection tags (bounding box regions) from the 'images' folder
    
    The script expects a .env file with:
    - TrainingEndpoint: URL of the Azure Custom Vision training API
    - TrainingKey: API key for authentication
    - ProjectID: The ID of the Custom Vision project for object detection
    
    The script also expects:
    - tagged-images.json: JSON file mapping image filenames to their tagged regions
    - images/: Folder containing the actual image files to upload
    """
    from dotenv import load_dotenv
    global training_client
    global custom_vision_project

    # Clear the console for a clean interface
    os.system('cls' if os.name=='nt' else 'clear')

    try:
        # ===== LOAD CONFIGURATION =====
        # Load environment variables from the .env file in the current directory
        load_dotenv()
        
        # Retrieve Azure Custom Vision settings from environment variables
        training_endpoint = os.getenv('TrainingEndpoint')  # Azure Custom Vision training API endpoint
        training_key = os.getenv('TrainingKey')            # API key for training service authentication
        project_id = os.getenv('ProjectID')                # The Custom Vision project ID for object detection

        # ===== AUTHENTICATION =====
        # Create authentication credentials using the training API key
        # The key will be included in the "Training-key" header of all API requests
        credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
        
        # Initialize the Custom Vision Training client
        # This client provides methods to interact with the Custom Vision training service
        training_client = CustomVisionTrainingClient(training_endpoint, credentials)

        # ===== RETRIEVE PROJECT =====
        # Fetch the Custom Vision project from Azure using its ID
        # The project contains all tags and images for this object detection model
        custom_vision_project = training_client.get_project(project_id)

        # ===== UPLOAD IMAGES =====
        # Call the Upload_Images function to upload and tag all images from the 'images' folder
        Upload_Images('images')
    except Exception as ex:
        # If any error occurs (authentication, network, file issues), print the error message
        print(ex)



def Upload_Images(folder):
    """
    Upload images with object detection tags to the Custom Vision project.
    
    This function:
    1. Reads tag definitions from the Custom Vision project
    2. Loads image metadata and bounding box coordinates from tagged-images.json
    3. Creates batch entries for each image with its regions (bounding boxes)
    4. Uploads the entire batch to Azure Custom Vision
    
    Parameters:
    - folder (str): Path to the folder containing the image files to upload
    
    Expected JSON format in tagged-images.json:
    {
        "files": [
            {
                "filename": "image1.jpg",
                "tags": [
                    {
                        "tag": "cat",
                        "left": 0.1,
                        "top": 0.2,
                        "width": 0.3,
                        "height": 0.4
                    }
                ]
            }
        ]
    }
    """
    print("Uploading images...")

    # ===== GET PROJECT TAGS =====
    # Retrieve all tag definitions from the Custom Vision project
    # Tags are the object categories we want to detect (e.g., "cat", "dog", "bird")
    # Each tag has a unique ID that we'll use when marking regions in images
    tags = training_client.get_tags(custom_vision_project.id)

    # ===== INITIALIZE IMAGE BATCH =====
    # Create an empty list to store image entries with their tagged regions
    # Each entry will contain an image and its bounding box coordinates for each detected object
    tagged_images_with_regions = []

    # ===== LOAD TAGGED IMAGES FROM JSON =====
    # Open and parse the JSON file containing image metadata and tag information
    # This JSON file maps each image to the objects it contains and their locations
    with open('tagged-images.json', 'r') as json_file:
        tagged_images = json.load(json_file)
        
        # Process each image in the JSON file
        for image in tagged_images['files']:
            # ===== EXTRACT IMAGE FILENAME =====
            # Get the filename of the image from the JSON
            file = image['filename']
            
            # ===== PROCESS TAGGED REGIONS =====
            # Initialize a list to store all the tagged regions (bounding boxes) for this image
            regions = []
            
            # Loop through each tag/object detected in this image
            for tag in image['tags']:
                # Get the tag name (object type) from the JSON, e.g., "apple", "banana"
                tag_name = tag['tag']
                
                # ===== LOOK UP TAG ID =====
                # Find the tag ID that corresponds to this tag name
                # Tag IDs are needed because the Azure API uses IDs, not names
                # The 'next()' function searches through the tags list to find a match
                tag_id = next(t for t in tags if t.name == tag_name).id
                
                # ===== CREATE REGION OBJECT =====
                # Create a Region object representing the bounding box for this detected object
                # The coordinates (left, top) and dimensions (width, height) are normalized values
                # (0.0 to 1.0, where 1.0 represents the full image dimension)
                # This allows the region to work with images of any size
                regions.append(Region(tag_id=tag_id, 
                                     left=tag['left'],      # X position of bounding box (0.0 to 1.0)
                                     top=tag['top'],        # Y position of bounding box (0.0 to 1.0)
                                     width=tag['width'],    # Width of bounding box (0.0 to 1.0)
                                     height=tag['height'])) # Height of bounding box (0.0 to 1.0)
            
            # ===== ADD IMAGE TO BATCH =====
            # Read the image file as binary data and create an ImageFileCreateEntry
            # This entry contains the image data and all its tagged regions
            with open(os.path.join(folder, file), mode="rb") as image_data:
                tagged_images_with_regions.append(ImageFileCreateEntry(name=file, 
                                                                        contents=image_data.read(), 
                                                                        regions=regions))

    # ===== UPLOAD BATCH =====
    # Send all the images with their tagged regions to the Custom Vision service as a batch
    # This is more efficient than uploading images one at a time
    # The API processes the batch and returns status information for each image
    upload_result = training_client.create_images_from_files(custom_vision_project.id, 
                                                             ImageFileCreateBatch(images=tagged_images_with_regions))
    
    # ===== CHECK UPLOAD STATUS =====
    # Verify whether the batch upload was successful
    # If any images failed to upload, we'll print detailed error information
    if not upload_result.is_batch_successful:
        # If the batch was not completely successful, print a failure message
        print("Image batch upload failed.")
        # Loop through each image in the result to show the status of individual uploads
        for image in upload_result.images:
            # Print the status of each image (e.g., "Success", "BadRequest", "Conflict")
            print("Image status: ", image.status)
    else:
        # If the batch was completely successful, print a success message
        print("Images uploaded.")

if __name__ == "__main__":
    """
    Script entry point.
    
    This ensures that main() is only executed when the script is run directly,
    not when it's imported as a module in another script.
    This is a Python best practice for creating reusable and testable scripts.
    """
    main()