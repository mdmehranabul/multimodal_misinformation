# Import necessary libraries
import pandas as pd
import google.generativeai as genai
import os
import time
from dotenv import load_dotenv
import requests
from PIL import Image
from io import BytesIO

# --- Configuration ---
# Load environment variables from the .env file
load_dotenv()

# Get your Gemini API key from the environment variable
# It is stored in a .env file for security.
API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the Gemini API with your key
genai.configure(api_key=API_KEY)

# The path to your input data file. Replace with your actual file path.
TEST_PATH  = "data/multimodal_test_public.tsv"
# The name of the output CSV file where results will be saved
OUTPUT_CSV_FILE = "gemini_output_2_way.csv"

# Choose your classification type: '2-way' or '6-way'
# The script will use the appropriate prompt and categories based on this setting.
CLASSIFICATION_TYPE = '2-way'  # Set to '2-way' or '6-way'

# --- Data Loading and Cleaning ---
def load_and_clean_data(path, sample_size=None, seed=42):
    """
    Loads and cleans data from a TSV file.
    
    This function expects a TSV file with columns including 'title' and 'image_url'.
    It filters for rows with images, removes duplicates, and maps labels.
    """
    print(f"Loading and cleaning data from {path}...")
    
    df = pd.read_csv(path, sep="\t")
    df['created_datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    df['hasImage'] = df['hasImage'].astype(str).str.upper() == 'TRUE'
    df = df[df['hasImage']]
    df = df.drop_duplicates(subset='id')

    df['2_way_label_name'] = df['2_way_label'].map({0: 'Misleading', 1: 'Real'})
    df['6_way_label_name'] = df['6_way_label'].map({
        0: 'Real', 1: 'Satire', 2: 'Misleading', 4: 'Manipulated',
        5: 'False Connection', 6: 'Imposter Content'
    })

    if sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    if 'title' not in df.columns or 'image_url' not in df.columns:
        raise ValueError("DataFrame must contain 'title' and 'image_url' columns.")
        
    return df

# --- Function to Classify Content ---
def classify_content_with_gemini(title, image_url):
    """
    Analyzes a title and image URL using the Gemini API and classifies them
    based on the CLASSIFICATION_TYPE setting.

    Args:
        title (str): The title of the content.
        image_url (str): The URL of the image.

    Returns:
        str: The classification result (e.g., '1: Real', '0: Misleading', etc.).
    """
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    if CLASSIFICATION_TYPE == '2-way':
        # Prompt for 2-way classification
        prompt = f"""
        Analyze the following title and image to determine if the title accurately describes the image.
        Classify the content as either 'Misleading' or 'Real'.

        - 'Real' means the title is an accurate and truthful description of the image content.
        - 'Misleading' means the title presents a false or inaccurate claim about the image.

        Provide your response in the exact format: "NUMBER: CATEGORY_NAME".
        Use the following categories:
        0: 'Misleading'
        1: 'Real'

        Title: "{title}"
        Image: [The image at the provided URL]
        """
    elif CLASSIFICATION_TYPE == '6-way':
        # Prompt for 6-way classification
        prompt = f"""
        Analyze the following title and image to determine the nature of the information.
        Classify the content into one of the following six categories:
        
        - **Real (0):** The title accurately and truthfully describes the image.
        - **Satire (1):** The title is humorous and not intended to be taken seriously.
        - **Misleading (2):** The title is factually true but used to frame the image in a deceptive way.
        - **Manipulated (3):** The image has been altered or doctored to deceive.
        - **False Connection (4):** The title and image are unrelated, but presented as if they are connected.
        - **Imposter Content (5):** The image and title impersonate a trusted source.
        
        Provide your response in the exact format: "NUMBER: CATEGORY_NAME".

        Title: "{title}"
        Image: [The image at the provided URL]
        """
    else:
        return "Error: Invalid CLASSIFICATION_TYPE. Please choose '2-way' or '6-way'."
        
    try:
        # Pass both the text prompt and the image URL to the model
        response = model.generate_content([prompt, image_url])
        # Return the clean, stripped text of the response
        return response.text.strip()
    except Exception as e:
        print(f"Error classifying content for title '{title}': {e}")
        # Return an error message for failed API calls
        return "Error"

# --- Main Script Execution ---
if __name__ == "__main__":
    if API_KEY is None:
        print("Error: Gemini API key not found. Please set it in your .env file.")
    else:
        # Record the start time
        start_time = time.time()
        try:
            # Load and clean the data using your defined function
            test_df = load_and_clean_data(TEST_PATH, sample_size=4000, seed=42)
            print(f"Successfully loaded and cleaned data with {len(test_df)} rows.")

            results = []
            # Loop through each row of the DataFrame and call the classification function
            for index, row in test_df.iterrows():
                title = row['title']
                image_url = row['image_url']
                print(f"[{index + 1}/{len(test_df)}] Classifying: '{title}'...")
                
                classification = classify_content_with_gemini(title, image_url)
                results.append(classification)
                
                # Add a small delay to avoid hitting rate limits
                time.sleep(1)
            
            # Add the classification results to a new column in the DataFrame
            test_df['gemini_classification'] = results
            
            # Save the updated DataFrame to a new CSV file
            test_df.to_csv(OUTPUT_CSV_FILE, index=False)
            
            # Record the end time
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"\nClassification complete! Results saved to '{OUTPUT_CSV_FILE}'.")
            print(f"Total time taken: {total_time:.2f} seconds.")
        
        except Exception as e:
            print(f"An unexpected error occurred during execution: {e}")