import requests
import json

# Load the access token from your key file
key_file_path = "/home/appuser/Grounded-SAM-2/key_file.json"
with open(key_file_path, "r") as f:
    key_info = json.load(f)

# Pick the inference key
access_token = key_info["inference"]["key"]

# Prepare the image file to send
image_path = "COCO/000000001000.jpg"
try:
    with open(image_path, "rb") as img_file:
        files = {"data": img_file}

        # Now use Authorization header with token
        headers = {"Authorization": f"Bearer {access_token}"}

        # Send the POST request
        response = requests.post(
            "http://127.0.0.1:8080/predictions/grounded_sam2_florence2",
            files=files,
            headers=headers,
            timeout=300,  # seconds
        )

        # Print the status code and response
        try:
            print(f"Status Code: {response.status_code}")
            print("Response JSON:", response.json())
        except Exception as e:
            print(f"❌ Failed to parse response as JSON: {e}")
            print("Raw Response Text:")
            print(response.text)

except FileNotFoundError:
    print(f"❌ Image file not found: {image_path}")
