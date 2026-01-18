# Import the dotenv library to load environment variables from a .env file
# This allows us to securely store credentials without hardcoding them in the source code
from dotenv import load_dotenv

# Import the os module for operating system operations (clearing console, reading environment variables)
import os

# Import the sys module to access command-line arguments passed to the script
# sys.argv[0] is the script name, sys.argv[1] onwards are user-provided arguments
import sys

# Import PIL (Python Imaging Library) components for image manipulation
# Image: used to open and work with image files
# ImageDraw: used to draw shapes (rectangles, polygons, etc.) on images
from PIL import Image, ImageDraw

# Import matplotlib's pyplot module for displaying and saving images
# This is used to visualize the annotated images with faces highlighted
from matplotlib import pyplot as plt

# Import Azure AI Vision Face API components for face detection and analysis
# FaceClient: The main client for communicating with Azure Face API
from azure.ai.vision.face import FaceClient

# Import models used for face detection configuration
# FaceDetectionModel: Specifies which detection algorithm to use
# FaceRecognitionModel: Specifies which recognition algorithm to use
# FaceAttributeTypeDetection01: Enum for available facial attributes to detect
from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel, FaceAttributeTypeDetection01

# Import credential handler for Azure API authentication
# AzureKeyCredential: Wraps the API key for secure authentication with Azure services
from azure.core.credentials import AzureKeyCredential


def main():
    """
    Main function that orchestrates the entire face detection and analysis workflow.
    This function handles:
    1. Loading configuration from environment variables
    2. Determining which image file to process
    3. Connecting to Azure Face API service
    4. Detecting faces and extracting their attributes
    5. Displaying results and creating annotated images
    """
    
    # Clear the console screen for a clean user interface
    # os.name=='nt' checks if running on Windows ('nt' = NT kernel)
    # Windows uses 'cls' command, Unix-like systems use 'clear' command
    os.system('cls' if os.name=='nt' else 'clear')

    try:
        # Load environment variables from .env file
        # This reads a local .env file containing sensitive credentials
        # The .env file should contain:
        #   - AI_SERVICE_ENDPOINT: URL of your Azure Face API resource
        #   - AI_SERVICE_KEY: Authentication key for the Face API
        load_dotenv()
        
        # Retrieve the Azure Face API endpoint from environment variables
        # Example format: https://eastus.api.cognitive.microsoft.com/
        cog_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        
        # Retrieve the authentication key from environment variables
        # This key is used to authenticate requests to the Azure Face API
        cog_key = os.getenv('AI_SERVICE_KEY')

        # Set default image file path
        # This image will be used if no command-line argument is provided
        image_file = 'images/face1.jpg'
        
        # Check if a command-line argument was provided by the user
        # len(sys.argv) > 1 means the user provided at least one argument
        # sys.argv[0] is always the script name, sys.argv[1] is the first user-provided argument
        # This allows users to run: python analyze-faces.py images/custom_image.jpg
        if len(sys.argv) > 1:
            # Override the default image file with the user-provided argument
            image_file = sys.argv[1]


        # Create and authenticate a Face API client
        # This client object will be used to send all face detection requests to Azure
        # Parameters explained:
        #   - endpoint: The base URL of the Azure Face API service for your region
        #   - credential: Wrapped API key that authenticates each request
        # The FaceClient communicates with Azure's servers to perform face detection
        face_client = FaceClient(
            endpoint=cog_endpoint,
            credential=AzureKeyCredential(cog_key))

        # Define which facial attributes we want Azure to detect and return
        # The Face API can detect many attributes; we specify only the ones we need
        # to reduce processing time and API call costs
        # Selected attributes:
        #   - HEAD_POSE: Detects the 3D orientation of the head (yaw, pitch, roll angles)
        #   - OCCLUSION: Detects if facial features are blocked (forehead, eyes, mouth)
        #   - ACCESSORIES: Detects items like glasses, hats, etc. worn on the face
        features = [FaceAttributeTypeDetection01.HEAD_POSE,
                    FaceAttributeTypeDetection01.OCCLUSION,
                    FaceAttributeTypeDetection01.ACCESSORIES]

        # Send image to Azure Face API for face detection and analysis
        # Open the image file in binary read mode ('rb') to get raw bytes
        # The Azure API expects binary image data, not a file path
        with open(image_file, mode="rb") as image_data:
            # Call the detect method on the Face API client
            # Parameters explained:
            #   - image_content: The binary image data to analyze
            #   - detection_model: DETECTION01 is the standard face detection model
            #   - recognition_model: RECOGNITION01 is used for recognizing face characteristics
            #   - return_face_id: False (we don't need unique face IDs for this exercise)
            #   - return_face_attributes: The list of attributes we want Azure to analyze
            # Returns: A list of detected faces with their attributes
            detected_faces = face_client.detect(
                image_content=image_data.read(),
                detection_model=FaceDetectionModel.DETECTION01,
                recognition_model=FaceRecognitionModel.RECOGNITION01,
                return_face_id=False,
                return_face_attributes=features,
            )

        # Initialize face counter to track and number detected faces
        face_count = 0
        
        # Check if any faces were detected in the image
        # If the list is not empty, at least one face was found
        if len(detected_faces) > 0:
            # Print the total number of faces detected
            print(len(detected_faces), 'faces detected.')
            
            # Loop through each detected face to extract and display its attributes
            for face in detected_faces:
                # Increment the face counter for this iteration
                face_count += 1
                
                # Print a header for the current face
                print('\nFace number {}'.format(face_count))
                
                # Print HEAD POSE information (3D head orientation)
                # Yaw: Left/right rotation (negative = looking left, positive = looking right)
                print(' - Head Pose (Yaw): {}'.format(face.face_attributes.head_pose.yaw))
                # Pitch: Up/down rotation (negative = looking down, positive = looking up)
                print(' - Head Pose (Pitch): {}'.format(face.face_attributes.head_pose.pitch))
                # Roll: Tilt rotation (negative = tilted left, positive = tilted right)
                print(' - Head Pose (Roll): {}'.format(face.face_attributes.head_pose.roll))
                
                # Print OCCLUSION information (facial feature blockages)
                # These are boolean values (True/False) indicating if features are blocked
                print(' - Forehead occluded?: {}'.format(face.face_attributes.occlusion["foreheadOccluded"]))
                print(' - Eye occluded?: {}'.format(face.face_attributes.occlusion["eyeOccluded"]))
                print(' - Mouth occluded?: {}'.format(face.face_attributes.occlusion["mouthOccluded"]))
                
                # Print ACCESSORIES information (objects worn on the face)
                print(' - Accessories:')
                # Loop through all detected accessories for this face
                for accessory in face.face_attributes.accessories:
                    # Print each accessory type (e.g., "glasses", "hat", "sunglasses")
                    print('   - {}'.format(accessory.type))
            
            # Call the annotation function to draw boxes around detected faces
            # This creates a visual representation of where faces were found
            annotate_faces(image_file, detected_faces)

    except Exception as ex:
        # Catch any errors that occurred during execution
        # This could include:
        #   - Authentication errors (invalid credentials)
        #   - File not found errors (image file doesn't exist)
        #   - Network errors (can't reach Azure service)
        #   - Invalid API responses
        # Print the error message to help debug the issue
        print(ex)

def annotate_faces(image_file, detected_faces):
    """
    Create a visual annotation of detected faces by drawing bounding boxes around them.
    This function takes the original image and the list of detected faces, then:
    1. Opens the original image file
    2. Draws rectangles around each face location
    3. Labels each face with a number
    4. Saves the annotated image to disk for visual inspection
    
    Args:
        image_file: Path to the original image file to annotate
        detected_faces: List of face objects returned by the Face API containing face rectangles
    """
    print('\nAnnotating faces in image...')

    # Create a matplotlib figure for displaying the image
    # figsize=(8, 6) sets the display size to 8x6 inches
    # This is independent of the actual image size
    fig = plt.figure(figsize=(8, 6))
    
    # Hide the axis labels and gridlines for a cleaner appearance
    # This shows just the image without matplotlib's default axis decorations
    plt.axis('off')
    
    # Open the original image file using PIL
    # Image.open() loads the image from disk into memory
    image = Image.open(image_file)
    
    # Create a drawing context for the image
    # ImageDraw.Draw() allows us to draw shapes on the image
    draw = ImageDraw.Draw(image)
    
    # Set the color for the bounding boxes
    # 'lightgreen' is a named color in PIL (RGB: 144, 238, 144)
    color = 'lightgreen'

    # Initialize face counter for numbering faces in the annotated image
    face_count = 0
    
    # Loop through each detected face to draw a bounding box around it
    for face in detected_faces:
        # Increment the face counter to assign a unique number to each face
        face_count += 1
        
        # Extract the face rectangle coordinates from the detected face
        # The face_rectangle contains:
        #   - left: X-coordinate of the left edge of the face
        #   - top: Y-coordinate of the top edge of the face
        #   - width: Width of the bounding box in pixels
        #   - height: Height of the bounding box in pixels
        r = face.face_rectangle
        
        # Calculate the bounding box coordinates for the rectangle
        # PIL's draw.rectangle() expects two points: top-left and bottom-right corners
        # Top-left corner: (left, top)
        # Bottom-right corner: (left + width, top + height)
        bounding_box = ((r.left, r.top), (r.left + r.width, r.top + r.height))
        
        # Redraw the drawing context (this is sometimes necessary for proper rendering)
        draw = ImageDraw.Draw(image)
        
        # Draw the rectangle (bounding box) around the detected face
        # Parameters:
        #   - bounding_box: The coordinates of the rectangle
        #   - outline: The color of the rectangle border (lightgreen)
        #   - width: The thickness of the border in pixels (5 pixels = thick border)
        draw.rectangle(bounding_box, outline=color, width=5)
        
        # Create a text label for this face (e.g., "Face number 1")
        annotation = 'Face number {}'.format(face_count)
        
        # Annotate the image with the face label
        # Parameters:
        #   - annotation: The text to display
        #   - (r.left, r.top): The position where the label appears (top-left of face)
        #   - backgroundcolor: The background color behind the text for visibility
        plt.annotate(annotation,(r.left, r.top), backgroundcolor=color)
    
    # Display the annotated image in the matplotlib window
    # plt.imshow() renders the image with all the rectangles we drew
    plt.imshow(image)
    
    # Define the output filename for the annotated image
    # The image will be saved in the current working directory
    outputfile = 'detected_faces.jpg'
    
    # Save the figure with all annotations to a JPEG file
    # fig.savefig() writes the current matplotlib figure to disk
    # The image includes the annotated image with bounding boxes and labels
    fig.savefig(outputfile)
    
    # Inform the user that the annotated image has been saved successfully
    # The output shows the filename where results can be found
    print(f'  Results saved in {outputfile}\n')


# Script entry point guard
# This ensures that main() only runs when the script is executed directly,
# not when it's imported as a module in another Python script
# How it works:
#   - If this file is run directly: __name__ is set to "__main__" by Python
#   - If this file is imported in another script: __name__ is set to the module name
# This pattern allows code to be reused as a module while still being executable
if __name__ == "__main__":
    # Call the main function to start the face detection and analysis workflow
    main()