# Import required modules for Azure Custom Vision prediction and file operations
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import os  # Used for environment variables and file/folder operations  # Used for environment variables and file/folder operations

def main():
    """
    Main entry point for the image classification testing script.
    
    This function:
    1. Loads configuration from environment variables
    2. Authenticates with the Azure Custom Vision prediction service
    3. Tests the trained model on images in the test-images folder
    4. Prints predictions for each image with confidence > 50%
    
    The script expects a .env file containing:
    - PredictionEndpoint: The Azure Custom Vision prediction API endpoint
    - PredictionKey: The API key for the prediction service
    - ProjectID: The ID of the Custom Vision project
    - ModelName: The name of the published model iteration to test
    """
    from dotenv import load_dotenv  # Load environment variables from .env file

    # Clear the console for a clean interface
    os.system('cls' if os.name=='nt' else 'clear')

    try:
        # ===== CONFIGURATION SETUP =====
        # Load environment variables from the .env file in the current directory
        load_dotenv()
        
        # Retrieve configuration settings from environment variables
        prediction_endpoint = os.getenv('PredictionEndpoint')  # Azure endpoint URL for predictions
        prediction_key = os.getenv('PredictionKey')  # API key for prediction authentication
        project_id = os.getenv('ProjectID')  # ID of the Custom Vision project
        model_name = os.getenv('ModelName')  # Name of the trained model to use for predictions  # Name of the trained model to use for predictions

        # ===== AUTHENTICATION =====
        # Create authentication credentials with the prediction key
        # This key will be included in the header of all prediction API requests
        credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        
        # Initialize the Custom Vision Prediction client with the endpoint and credentials
        # This client is used to make predictions on images using the trained model
        prediction_client = CustomVisionPredictionClient(endpoint=prediction_endpoint, credentials=credentials)

        # ===== IMAGE CLASSIFICATION =====
        # Process each test image in the test-images folder
        for image in os.listdir('test-images'):
            # Read the image file as binary data
            image_path = os.path.join('test-images', image)
            image_data = open(image_path, "rb").read()
            
            # Send the image to the trained model for classification
            # The model analyzes the image and returns predictions for each category
            # (e.g., apple, banana, oranges with confidence percentages)
            results = prediction_client.classify_image(project_id, model_name, image_data)

            # ===== RESULTS PROCESSING =====
            # Loop over each predicted label returned by the model
            for prediction in results.predictions:
                # Only print predictions with confidence > 50%
                # This filters out low-confidence guesses for cleaner output
                # prediction.probability is a decimal (0.0 to 1.0)
                if prediction.probability > 0.5:
                    # Print the image filename, predicted category, and confidence percentage
                    # Format: image.jpg : apple (85%)
                    print(image, ': {} ({:.0%})'.format(prediction.tag_name, prediction.probability))
    except Exception as ex:
        # If any error occurs during prediction, print it for debugging
        print(ex)

if __name__ == "__main__":
    """
    Script entry point.
    
    This ensures that main() is only executed when the script is run directly,
    not when it's imported as a module in another script.
    This is a Python best practice for creating reusable scripts.
    """
    main()

