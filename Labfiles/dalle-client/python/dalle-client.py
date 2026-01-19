import os  # For file and directory operations
import json  # For parsing JSON responses from the API

# Add references
# Azure authentication and OpenAI client for DALL-E image generation
from dotenv import load_dotenv  # Loads environment variables from .env file
from azure.identity import DefaultAzureCredential, get_bearer_token_provider  # Azure authentication
from openai import AzureOpenAI  # OpenAI client configured for Azure
import requests  # For downloading generated images


def main():
    """
    Main entry point for the DALL-E image generation application.
    This function:
    1. Loads configuration from environment variables
    2. Authenticates with Azure using DefaultAzureCredential
    3. Creates an OpenAI client preconfigured for Azure
    4. Loops to accept user prompts and generate images
    5. Saves generated images to disk
    """

    # Clear the console screen for a clean UI
    # os.name == 'nt' checks if running on Windows (NT = New Technology)
    # If Windows: use 'cls' command, otherwise (Linux/Mac): use 'clear' command
    os.system('cls' if os.name=='nt' else 'clear')
        
    try:
        # =============================================================================
        # STEP 1: LOAD CONFIGURATION FROM ENVIRONMENT VARIABLES
        # =============================================================================
        # load_dotenv() reads the .env file and loads variables into os.getenv()
        # This keeps sensitive information (like API endpoints and keys) out of source code
        load_dotenv()
        
        # Retrieve the Azure OpenAI endpoint URL for DALL-E
        # This is the base URL where the DALL-E service is hosted
        endpoint = os.getenv("ENDPOINT")
        
        # Retrieve the name of the deployed DALL-E model
        # The deployment name identifies which DALL-E version to use (e.g., 'dall-e-3')
        model_deployment = os.getenv("MODEL_DEPLOYMENT")
        
        # Retrieve the API version to use for requests
        # Different API versions may have different features and response formats
        api_version = os.getenv("API_VERSION")
        
        # =============================================================================
        # STEP 2: AUTHENTICATE WITH AZURE AND CREATE OPENAI CLIENT
        # =============================================================================
        # Create a token provider that uses Azure credentials for authentication
        # DefaultAzureCredential() automatically handles authentication using:
        # - Environment variables
        # - Managed identity
        # - Interactive login if needed
        # The exclude_* parameters prioritize interactive login
        # The scope "https://cognitiveservices.azure.com/.default" authorizes access to Azure AI services
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(
                exclude_environment_credential=True,
                exclude_managed_identity_credential=True
            ), 
            "https://cognitiveservices.azure.com/.default"
        )
        
        # Create the Azure OpenAI client preconfigured for this service
        # This client handles all communication with the DALL-E model
        # Parameters:
        # - api_version: Specifies which Azure OpenAI API version to use
        # - azure_endpoint: Base URL for the service
        # - azure_ad_token_provider: Function that provides authentication tokens
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider
        )
        
        # =============================================================================
        # STEP 3: INITIALIZE IMAGE COUNTER AND START INTERACTIVE LOOP
        # =============================================================================
        # Counter to track how many images have been generated in this session
        # Increments with each image to create unique filenames (image_1.png, image_2.png, etc.)
        img_no = 0
        
        # Loop until the user types 'quit'
        # This creates an interactive session where users can request multiple images
        while True:
            # Prompt the user for an image description
            # The prompt should describe what they want the DALL-E model to generate
            input_text = input("Enter the prompt (or type 'quit' to exit): ")
            
            # Check if user wants to exit the application
            if input_text.lower() == "quit":
                break
            
            # Check if user entered an empty prompt
            # Empty prompts would fail, so we ask for input again
            if len(input_text) == 0:
                print("Please enter a prompt.")
                continue
            
            # =============================================================================
            # STEP 4: SEND PROMPT TO DALL-E MODEL AND RETRIEVE GENERATED IMAGE URL
            # =============================================================================
            # Send the user's prompt to the DALL-E model and request image generation
            # Parameters:
            # - model: Specifies which deployed DALL-E model to use
            # - prompt: The user's description of the image to generate
            # - n: Number of images to generate (1 in this case)
            result = client.images.generate(
                model=model_deployment,
                prompt=input_text,
                n=1
            )
            
            # Convert the result object to JSON format for easier data access
            # The model_dump_json() method serializes the response object to JSON string
            # json.loads() parses the JSON string into a Python dictionary
            json_response = json.loads(result.model_dump_json())
            
            # Extract the URL of the generated image from the response
            # json_response["data"][0]["url"] navigates to:
            # - "data": Array of generated images
            # - [0]: First (and only) image in the array
            # - "url": The URL where the generated image is hosted
            image_url = json_response["data"][0]["url"]

            # =============================================================================
            # STEP 5: SAVE THE GENERATED IMAGE
            # =============================================================================
            # Increment the image counter for the next image
            img_no += 1
            
            # Create a filename for the generated image
            # Format: image_1.png, image_2.png, etc.
            file_name = f"image_{img_no}.png"
            
            # Call the save_image function to download and save the image locally
            # Parameters:
            # - image_url: The URL of the generated image (from DALL-E)
            # - file_name: The filename to save it as
            save_image(image_url, file_name)


    # Error handling: Catch and display any exceptions that occur
    # This could include network errors, authentication failures, API errors, etc.
    except Exception as ex:
        print(ex)


def save_image(image_url, file_name):
    """
    Downloads an image from a URL and saves it to disk.
    
    Parameters:
    - image_url: URL of the image to download (provided by DALL-E)
    - file_name: Filename to save the image as (should be .png)
    
    This function:
    1. Creates an 'images' directory if it doesn't exist
    2. Downloads the image from the URL
    3. Saves the image to the images folder
    4. Prints confirmation message
    """
    
    # =============================================================================
    # CREATE IMAGES DIRECTORY IF IT DOESN'T EXIST
    # =============================================================================
    # Set the directory path for storing generated images
    # os.path.join() combines the current working directory with 'images' folder name
    image_dir = os.path.join(os.getcwd(), 'images')

    # Check if the images directory already exists
    # os.path.isdir() returns True if the path is an existing directory
    if not os.path.isdir(image_dir):
        # Create the 'images' directory if it doesn't exist
        # os.mkdir() creates a single new directory
        os.mkdir(image_dir)

    # =============================================================================
    # DOWNLOAD AND SAVE THE IMAGE
    # =============================================================================
    # Create the full file path by combining the images directory and filename
    image_path = os.path.join(image_dir, file_name)

    # Download the image from the URL
    # requests.get() fetches the image content
    # .content returns the binary image data (not text)
    generated_image = requests.get(image_url).content
    
    # Open a file in binary write mode ('wb') and save the image
    # 'wb' mode writes binary data (the image file)
    # The 'with' statement ensures the file is properly closed after writing
    with open(image_path, "wb") as image_file:
        # Write the downloaded image data to the file
        image_file.write(generated_image)
    
    # Print confirmation message showing where the image was saved
    print(f"Image saved as {image_path}")


# This guard ensures the main() function only runs when the script is executed directly
# It doesn't run if this file is imported as a module in another script
if __name__ == '__main__': 
    main()