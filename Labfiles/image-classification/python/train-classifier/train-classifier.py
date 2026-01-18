# Import required modules for Azure Custom Vision and file operations
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import time  # Used for delays during model training polling
import os  # Used for environment variables and file/folder operations

# Global variables that will be set during initialization
# These store the Azure client and project information needed throughout the script
training_client = None
custom_vision_project = None

def main():
    """
    Main entry point for the image classification training script.
    
    This function orchestrates the entire training workflow:
    1. Loads environment variables from a .env file
    2. Authenticates with the Azure Custom Vision service
    3. Connects to an existing Custom Vision project
    4. Uploads training images with their tags
    5. Trains the classification model
    
    The script expects a .env file containing:
    - TrainingEndpoint: The Azure Custom Vision training API endpoint
    - TrainingKey: The API key for authentication
    - ProjectID: The ID of the existing Custom Vision project to use
    """
    from dotenv import load_dotenv  # Load environment variables from .env file
    global training_client
    global custom_vision_project

    # Clear the console for a clean interface
    os.system('cls' if os.name=='nt' else 'clear')

    try:
        # ===== CONFIGURATION SETUP =====
        # Load environment variables from the .env file in the current directory
        load_dotenv()
        
        # Retrieve configuration settings from environment variables
        training_endpoint = os.getenv('TrainingEndpoint')  # Azure endpoint URL
        training_key = os.getenv('TrainingKey')  # API key for authentication
        project_id = os.getenv('ProjectID')  # ID of the project to train

        # ===== AUTHENTICATION =====
        # Create authentication credentials with the training key
        # This key will be included in the header of all API requests
        credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
        
        # Initialize the Custom Vision Training client with the endpoint and credentials
        # This client is used to communicate with Azure's Custom Vision service
        training_client = CustomVisionTrainingClient(training_endpoint, credentials)

        # ===== PROJECT RETRIEVAL =====
        # Fetch the Custom Vision project using the project ID
        # This verifies the connection and retrieves project details
        custom_vision_project = training_client.get_project(project_id)

        # ===== TRAINING WORKFLOW =====
        # Upload training images from the specified folder
        # The folder should contain subfolders named after each class/tag
        Upload_Images('more-training-images')

        # Train the model using the uploaded images
        # This sends the project to Azure for training and monitors progress
        Train_Model()
        
    except Exception as ex:
        # If any error occurs, print it for debugging
        print(ex)

def Upload_Images(folder):
    """
    Upload training images to the Custom Vision project with appropriate tags.
    
    This function:
    1. Retrieves all tags (categories/classes) defined in the project
    2. For each tag, finds a corresponding subfolder in the training data
    3. Reads each image in the subfolder
    4. Uploads the image to Azure with the tag associated
    
    Args:
        folder (str): Path to the root folder containing training images.
                     This folder should have subfolders named after each tag/class.
                     For example: 'training-images/apple/', 'training-images/banana/'
    
    The folder structure should be:
    more-training-images/
    ├── apple/          (subfolder named after the tag)
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── banana/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── orange/
        └── ...
    """
    print("Uploading images...")
    
    # Get all tags (categories) that exist in the Custom Vision project
    # Tags must be pre-created in the project before uploading images
    tags = training_client.get_tags(custom_vision_project.id)
    
    # Iterate through each tag/category
    for tag in tags:
        print(tag.name)  # Print the tag name for progress tracking
        
        # Get the path to the folder containing images for this tag
        tag_folder_path = os.path.join(folder, tag.name)
        
        # Process each image file in this tag's folder
        for image in os.listdir(tag_folder_path):
            # Read the image file as binary data
            image_path = os.path.join(tag_folder_path, image)
            image_data = open(image_path, "rb").read()
            
            # Upload the image to the project with this tag
            # The tag.id links the image to the correct category
            training_client.create_images_from_data(custom_vision_project.id, image_data, [tag.id])

def Train_Model():
    """
    Train the Custom Vision model using the uploaded images.
    
    This function:
    1. Initiates a training iteration on the Custom Vision project
    2. Polls the training status periodically
    3. Waits until training is complete
    4. Reports completion to the user
    
    During training, Azure's machine learning processes the tagged images
    to learn the visual characteristics of each category (apple, banana, etc.)
    
    Note: Training can take several minutes depending on the number of images.
          The function checks status every 5 seconds to update the user.
    """
    print("Training ...")
    
    # Send the project to Azure for training
    # This initiates a machine learning process using the uploaded images
    iteration = training_client.train_project(custom_vision_project.id)
    
    # Keep checking the training status until it's complete
    # Training may take several minutes, so we poll periodically
    while (iteration.status != "Completed"):
        # Fetch the latest status of the current training iteration
        iteration = training_client.get_iteration(custom_vision_project.id, iteration.id)
        
        # Print the current status (e.g., "Training", "Validating")
        print(iteration.status, '...')
        
        # Wait 5 seconds before checking status again
        # This prevents overwhelming the Azure API with requests
        time.sleep(5)
    
    # When the loop exits, training is complete
    print("Model trained!")


if __name__ == "__main__":
    """
    Script entry point.
    This ensures that main() is only executed when the script is run directly,
    not when it's imported as a module in another script.
    This is a Python best practice for creating reusable scripts.
    """
    main()


