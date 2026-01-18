# Import Azure Custom Vision for making predictions on a trained model
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
# Import authentication credentials for API calls
from msrest.authentication import ApiKeyCredentials
# Import matplotlib for creating visualizations and displaying annotated images
from matplotlib import pyplot as plt
# Import PIL (Pillow) for image manipulation - Image for loading, ImageDraw for drawing boxes/annotations
from PIL import Image, ImageDraw, ImageFont
# Import NumPy for numerical operations (getting image dimensions)
import numpy as np
# Import os for system operations (clearing console, getting environment variables)
import os

def main():
    """
    Main function that orchestrates the object detection workflow.
    This function:
    1. Loads environment configuration from a .env file
    2. Authenticates with Azure Custom Vision service
    3. Reads an image file and sends it to the trained model
    4. Processes predictions and displays results with confidence scores
    5. Creates an annotated image with bounding boxes around detected objects
    """
    from dotenv import load_dotenv

    # Clear the console screen for a clean start
    # os.name == 'nt' checks if running on Windows (NT = New Technology)
    os.system('cls' if os.name=='nt' else 'clear')

    try:
        # =============================================================================
        # STEP 1: LOAD CONFIGURATION SETTINGS
        # =============================================================================
        # load_dotenv() reads the .env file and loads environment variables into os.getenv()
        # This keeps sensitive credentials (like API keys) out of the source code
        load_dotenv()
        
        # Retrieve Azure Custom Vision service configuration from environment variables
        # These values should be set in your .env file (not included in version control)
        prediction_endpoint = os.getenv('PredictionEndpoint')  # Azure service endpoint URL
        prediction_key = os.getenv('PredictionKey')            # API authentication key
        project_id = os.getenv('ProjectID')                    # Unique ID of your trained model project
        model_name = os.getenv('ModelName')                    # Name of the specific iteration/version of your model

        # =============================================================================
        # STEP 2: AUTHENTICATE WITH AZURE CUSTOM VISION SERVICE
        # =============================================================================
        # Create credentials object with the API key - this is used for all API requests
        # The header "Prediction-key" tells Azure which API key to validate
        credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        
        # Initialize the prediction client with endpoint and credentials
        # This client object will handle communication with the Azure Custom Vision service
        prediction_client = CustomVisionPredictionClient(endpoint=prediction_endpoint, credentials=credentials)

        # =============================================================================
        # STEP 3: LOAD IMAGE AND SEND TO MODEL FOR OBJECT DETECTION
        # =============================================================================
        # Specify the image file to analyze (should be in the same directory as this script)
        image_file = 'produce.jpg'
        print('Detecting objects in', image_file)
        
        # Open the image file in binary read mode ('rb') - required for API transmission
        # The 'with' statement ensures the file is properly closed after reading
        with open(image_file, mode="rb") as image_data:
            # Send the image to the Azure Custom Vision prediction service
            # Returns a results object containing all detected objects and their confidence scores
            results = prediction_client.detect_image(project_id, model_name, image_data)

        # =============================================================================
        # STEP 4: PROCESS AND DISPLAY DETECTION RESULTS
        # =============================================================================
        # Iterate through each object detected in the image
        for prediction in results.predictions:
            # Filter predictions to only show confident detections (>50% probability)
            # probability is expressed as a decimal (0.0 to 1.0), so multiply by 100 for percentage
            if (prediction.probability*100) > 50:
                # Print the detected object class name (e.g., "apple", "banana", "orange")
                print(prediction.tag_name)

        # =============================================================================
        # STEP 5: CREATE ANNOTATED IMAGE WITH BOUNDING BOXES
        # =============================================================================
        # Call the save_tagged_images function to draw boxes around detected objects
        # and save the annotated image to a file
        save_tagged_images(image_file, results.predictions)

    except Exception as ex:
        # Catch and print any errors that occur during execution
        # This could include file not found, authentication errors, network issues, etc.
        print(ex)

def save_tagged_images(source_path, detected_objects):
    """
    Create a visual representation of detected objects with bounding boxes and labels.
    
    Parameters:
    - source_path: Path to the original image file to annotate
    - detected_objects: List of detected object predictions from the model
                       Each contains: tag_name, probability, and bounding_box coordinates
    
    This function:
    1. Loads the image and gets its dimensions
    2. Creates a visualization figure for display
    3. Draws rectangles around each detected object (bounding boxes)
    4. Adds labels with object names and confidence percentages
    5. Saves the annotated image to a file
    """
    
    # Load the image using Pillow (PIL) and convert to NumPy array for dimension extraction
    image = Image.open(source_path)
    
    # Get image dimensions: height (h), width (w), and channels (ch)
    # np.array(image).shape returns (height, width, channels) for RGB images
    h, w, ch = np.array(image).shape
    
    # Create a matplotlib figure for visualization
    # figsize=(8, 8) creates an 8x8 inch figure
    fig = plt.figure(figsize=(8, 8))
    
    # Turn off axis labels and tick marks for cleaner visualization
    plt.axis('off')

    # Create a drawing object to annotate the image
    # ImageDraw allows us to draw shapes like lines and rectangles on the image
    draw = ImageDraw.Draw(image)
    
    # Calculate line width proportional to image width
    # Dividing by 100 ensures the border is visible but not overwhelming regardless of image size
    lineWidth = int(w/100)
    
    # Set the color for bounding boxes and labels
    color = 'magenta'
    
    # Iterate through each detected object to draw its bounding box and label
    for detected_object in detected_objects:
        # Only show objects with a confidence > 50%
        # probability is a decimal (0.0 to 1.0), so multiply by 100 to get percentage
        if (detected_object.probability*100) > 50:
            # Convert proportional bounding box coordinates to absolute pixel coordinates
            # Azure returns normalized values (0.0 to 1.0) where:
            # - (0,0) is top-left corner, (1,1) is bottom-right corner
            # We multiply by image dimensions to convert to actual pixel positions
            left = detected_object.bounding_box.left * w 
            top = detected_object.bounding_box.top * h 
            height = detected_object.bounding_box.height * h
            width = detected_object.bounding_box.width * w
            
            # Define the four corners of the bounding box rectangle
            # Format: ((x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1))
            # The last point closes the rectangle back to the starting point
            points = (
                (left, top),                          # Top-left corner
                (left+width, top),                    # Top-right corner
                (left+width, top+height),             # Bottom-right corner
                (left, top+height),                   # Bottom-left corner
                (left, top)                           # Close back to top-left
            )
            
            # Draw the bounding box as a closed line/rectangle
            draw.line(points, fill=color, width=lineWidth)
            
            # Add text label above the bounding box with object name and confidence percentage
            # Format: "apple: 95.23%" (tag_name and probability formatted to 2 decimal places)
            label_text = detected_object.tag_name + ": {0:.2f}%".format(detected_object.probability * 100)
            plt.annotate(label_text, (left, top), backgroundcolor=color)
    
    # Display the annotated image in the figure
    plt.imshow(image)
    
    # Define the output filename for the annotated image
    outputfile = 'output.jpg'
    
    # Save the figure with annotations to a file
    fig.savefig(outputfile)
    
    # Confirm to the user that the image has been saved
    print('Results saved in', outputfile)


if __name__ == "__main__":
    main()
