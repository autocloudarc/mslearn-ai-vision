# Standard library imports for system and file operations
import os  # Provides access to operating system functions (e.g., clearing console)
from urllib.request import urlopen, Request  # For downloading images from URLs
import base64  # For encoding binary image data to text format for API transmission
from pathlib import Path  # For handling file paths in a cross-platform way
from dotenv import load_dotenv  # Loads environment variables from .env file

# Add references
# Azure AI and authentication imports for connecting to generative AI services
from azure.identity import DefaultAzureCredential  # Handles Azure authentication automatically
from azure.ai.projects import AIProjectClient  # Client for Azure AI Foundry projects
from openai import AzureOpenAI  # OpenAI client configured for Azure deployment


def main():
    """
    Main entry point for the vision-enabled chat application.
    This function:
    1. Connects to Azure AI Foundry project with a vision-capable model
    2. Sets up a chat client for multimodal (text + image) interactions
    3. Launches an interactive loop for users to ask questions about images
    4. Sends prompts with images to the model and displays responses
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
        # This keeps sensitive information (like API keys) out of source code
        load_dotenv()
        
        # Retrieve the Azure AI Foundry project endpoint
        # This is the base URL for all API calls to your project
        project_endpoint = os.getenv("PROJECT_CONNECTION")
        
        # Retrieve the name of the deployed model to use for inference
        # The model deployment name (e.g., 'gpt-4o') identifies which model to call
        model_deployment = os.getenv("MODEL_DEPLOYMENT")

        # =============================================================================
        # STEP 2: AUTHENTICATE WITH AZURE AND CREATE PROJECT CLIENT
        # =============================================================================
        # DefaultAzureCredential() automatically handles authentication using:
        # - Environment variables
        # - System credentials
        # - Interactive login if needed
        # The exclude_* parameters prevent using certain credential types
        # to prioritize other authentication methods
        
        # Create a client object to interact with the Azure AI Foundry project
        # This client provides access to models and services within the project
        project_client = AIProjectClient(
            credential=DefaultAzureCredential(
                exclude_environment_credential=True,
                exclude_managed_identity_credential=True
            ),
            endpoint=project_endpoint,
        )

        
        # =============================================================================
        # STEP 3: CREATE OPENAI CLIENT FOR CHAT COMPLETIONS
        # =============================================================================
        # get_openai_client() returns an OpenAI client preconfigured for this project
        # api_version specifies which version of the OpenAI API to use
        # The client handles communication with the deployed model
        openai_client = project_client.get_openai_client(api_version="2024-10-21")
        



# =============================================================================
        # STEP 4: INITIALIZE SYSTEM MESSAGE AND START INTERACTIVE LOOP
        # =============================================================================
        # The system message defines the AI's behavior and personality
        # It's sent with every request to guide how the model responds
        system_message = "You are an AI assistant in a grocery store that sells fruit. You provide detailed answers to questions about produce."
        
        # Initialize the user prompt variable
        # This will store the user's question about the image
        prompt = ""

        # Loop until the user types 'quit'
        # This creates an interactive chat session
        while True:
            # Prompt the user for input
            # They can ask questions about the fruit image shown to the model
            prompt = input("\nAsk a question about the image\n(or type 'quit' to exit)\n")
            
            # Check if user wants to exit
            if prompt.lower() == "quit":
                break
            
            # Check if user entered an empty prompt
            elif len(prompt) == 0:
                print("Please enter a question.\n")
            
            # User entered a valid prompt
            else:
                print("Getting a response ...\n")

                # =============================================================================
                # STEP 5: ENCODE IMAGE AND SEND TO MODEL WITH PROMPT
                # =============================================================================
                # The image URL points to a fruit image hosted on GitHub
                # We'll download, encode, and send this to the model along with the user's question
                # Get a response to image input
                script_dir = Path(__file__).parent  # Get the directory of the script
                image_path = script_dir / 'mystery-fruit.jpeg'
                mime_type = "image/jpeg"

                # Read and encode the image file
                with open(image_path, "rb") as image_file:
                    base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

                # Include the image file data in the prompt
                data_url = f"data:{mime_type};base64,{base64_encoded_data}"
                response = openai_client.chat.completions.create(
                        model=model_deployment,
                        messages=[
                            {"role": "system", "content": system_message},
                            { "role": "user", "content": [  
                                { "type": "text", "text": prompt},
                                { "type": "image_url", "image_url": {"url": data_url}}
                            ] } 
                        ]
                )
                print(response.choices[0].message.content)
                    


    # Error handling: Catch and display any exceptions that occur
    # This could include network errors, authentication failures, or API errors
    except Exception as ex:
        print(ex)

# This guard ensures the main() function only runs when the script is executed directly
# It doesn't run if this file is imported as a module in another script
if __name__ == '__main__': 
    main()