# Import required libraries for environment variable management, system operations, and image processing
from dotenv import load_dotenv  # Load environment variables from .env file
import os  # Operating system operations (console clearing, environment variable access)
import time  # Time module (currently imported but not used)
import sys  # System-specific parameters and functions (command line argument handling)
from PIL import Image, ImageDraw  # PIL/Pillow for image manipulation and drawing
from matplotlib import pyplot as plt  # Matplotlib for image visualization and saving

# Import Azure AI Vision libraries for optical character recognition (OCR)
from azure.ai.vision.imageanalysis import ImageAnalysisClient  # Main client for Azure AI Vision
from azure.ai.vision.imageanalysis.models import VisualFeatures  # Enum for selecting analysis features
from azure.core.credentials import AzureKeyCredential  # Credentials handler for Azure API key authentication


def main():
    """
    Main function to orchestrate the OCR (Optical Character Recognition) workflow.
    Reads text from an image file using Azure AI Vision service and annotates the results.
    """
    
    # Clear the console for a clean display
    # Use 'cls' command for Windows (os.name=='nt') or 'clear' for Unix-like systems
    os.system('cls' if os.name=='nt' else 'clear')

    try:
        # Load configuration settings from environment variables
        # load_dotenv() reads the .env file in the current directory
        load_dotenv()
        
        # Retrieve Azure AI Vision service credentials from environment variables
        # AI_SERVICE_ENDPOINT: The base URL for the Azure AI Vision service (e.g., https://xxxxx.cognitiveservices.azure.com/)
        # AI_SERVICE_KEY: The authentication key/API key for the Azure AI Vision service
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        # Determine which image file to process
        # Default to 'images/Lincoln.jpg' if no command-line argument is provided
        image_file = 'images/Lincoln.jpg'
        
        # Check if a command-line argument was provided
        # sys.argv[0] is the script name, sys.argv[1] is the first argument after the script name
        # This allows users to run: python read-text.py custom-image.jpg
        if len(sys.argv) > 1:
            image_file = sys.argv[1]


        # Create and authenticate the Azure AI Vision client
        # This client will be used to send image analysis requests to the Azure service
        # Parameters:
        #   - endpoint: The base URL of your Azure AI Vision resource
        #   - credential: The API key wrapped in AzureKeyCredential for authentication
        cv_client = ImageAnalysisClient(
            endpoint=ai_endpoint,
            credential=AzureKeyCredential(ai_key))
        
        # Read and prepare the image file for analysis
        # Open the image file in binary read mode ('rb') to get raw bytes
        with open(image_file, "rb") as f:
            image_data = f.read()
        
        # Inform the user which image is being processed
        print (f"\nReading text in {image_file}")

        # Send the image to Azure AI Vision service for text recognition (OCR)
        # Parameters:
        #   - image_data: The binary image data to analyze
        #   - visual_features: List of analysis features to perform (only READ/OCR in this case)
        # The READ feature performs Optical Character Recognition (OCR) on the image
        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[VisualFeatures.READ])

        # Extract and display the recognized text lines from the analysis result
        # Check if the READ result contains data (text was successfully detected)
        if result.read is not None:
            print("\nText:")
            
            # Iterate through each line of text detected in the image
            # result.read.blocks[0] gets the first text block
            # .lines contains all the detected text lines within that block
            for line in result.read.blocks[0].lines:
                print(f" {line.text}")  # Print each complete line of text        
            # Draw bounding boxes around detected text lines on the image
            # This creates a visual annotation showing where text was found
            annotate_lines(image_file, result.read)

            # Extract and display individual words along with their confidence scores
            # This provides more granular detail about what was recognized
            print ("\nIndividual words:")
            
            # Iterate through each detected line
            for line in result.read.blocks[0].lines:
                # Iterate through each word within the line
                for word in line.words:
                    # Print each word and its confidence score (0.00 to 1.00, shown as percentage)
                    # Higher confidence indicates the OCR engine is more certain about the recognition
                    print(f"  {word.text} (Confidence: {word.confidence:.2f}%)")
            
            # Draw bounding boxes around individual detected words on the image
            # This creates a more detailed visual annotation at the word level
            annotate_words(image_file, result.read)

    except Exception as ex:
        # Catch and print any errors that occur during execution
        # This could include authentication errors, file not found, network issues, etc.
        print(ex)

def annotate_lines(image_file, detected_text):
    """
    Create a visual annotation of detected text lines by drawing bounding polygons.
    
    Args:
        image_file: Path to the original image file
        detected_text: The read result from Azure AI Vision containing detected lines
    """
    print(f'\nAnnotating lines of text in image...')

    # Load the original image file for annotation
    image = Image.open(image_file)
    
    # Create a matplotlib figure with dimensions matching the image (converted from pixels to inches at 100 DPI)
    fig = plt.figure(figsize=(image.width/100, image.height/100))
    
    # Hide the axis labels and ticks for a cleaner image output
    plt.axis('off')
    
    # Create a drawing context to add annotations to the image
    draw = ImageDraw.Draw(image)
    
    # Set the color for the bounding box outlines (cyan = light blue)
    color = 'cyan'

    # Iterate through each detected text line
    for line in detected_text.blocks[0].lines:
        # Extract the bounding polygon coordinates for this line
        # A bounding polygon is defined by 4 corner points (x, y coordinates)
        r = line.bounding_polygon
        
        # Convert the polygon points to a tuple format expected by PIL's draw.polygon()
        # Each point is extracted as (x, y) coordinate pair
        rectangle = ((r[0].x, r[0].y),(r[1].x, r[1].y),(r[2].x, r[2].y),(r[3].x, r[3].y))
        
        # Draw the bounding polygon on the image
        # Parameters: polygon outline color, line width of 3 pixels
        draw.polygon(rectangle, outline=color, width=3)

    # Display the annotated image
    plt.imshow(image)
    
    # Remove extra padding around the image
    plt.tight_layout(pad=0)
    
    # Set the output filename for the annotated image
    textfile = 'lines.jpg'
    
    # Save the annotated image to disk
    fig.savefig(textfile)
    
    # Inform the user where the results were saved
    print('  Results saved in', textfile)
    
def annotate_words(image_file, detected_text):
    """
    Create a visual annotation of detected individual words by drawing bounding polygons.
    This provides more granular detail than line-level annotation.
    
    Args:
        image_file: Path to the original image file
        detected_text: The read result from Azure AI Vision containing detected words
    """
    print(f'\nAnnotating individual words in image...')

    # Load the original image file for annotation
    image = Image.open(image_file)
    
    # Create a matplotlib figure with dimensions matching the image (converted from pixels to inches at 100 DPI)
    fig = plt.figure(figsize=(image.width/100, image.height/100))
    
    # Hide the axis labels and ticks for a cleaner image output
    plt.axis('off')
    
    # Create a drawing context to add annotations to the image
    draw = ImageDraw.Draw(image)
    
    # Set the color for the bounding box outlines (cyan = light blue)
    color = 'cyan'

    # Nested loop: Iterate through each detected line, then each word within that line
    for line in detected_text.blocks[0].lines:
        for word in line.words:
            # Extract the bounding polygon coordinates for this word
            # A bounding polygon is defined by 4 corner points (x, y coordinates)
            r = word.bounding_polygon
            
            # Convert the polygon points to a tuple format expected by PIL's draw.polygon()
            # Each point is extracted as (x, y) coordinate pair
            rectangle = ((r[0].x, r[0].y),(r[1].x, r[1].y),(r[2].x, r[2].y),(r[3].x, r[3].y))
            
            # Draw the bounding polygon on the image around each word
            # Parameters: polygon outline color, line width of 3 pixels
            draw.polygon(rectangle, outline=color, width=3)

    # Display the annotated image
    plt.imshow(image)
    
    # Remove extra padding around the image
    plt.tight_layout(pad=0)
    
    # Set the output filename for the annotated image
    textfile = 'words.jpg'
    
    # Save the annotated image to disk
    fig.savefig(textfile)
    
    # Inform the user where the results were saved
    print('  Results saved in', textfile)



# Script entry point - ensures main() only runs when script is executed directly,
# not when it's imported as a module in another script
if __name__ == "__main__":
    main()
