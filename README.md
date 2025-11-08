# ComfyUI Cheap Trainer Utils

ComfyUI nodes for managing LoRA training workflows with Google Drive. Built for temporary cloud instances where you need to save training outputs before the instance shuts down.

## What It Does

- Loads image-caption pairs from Google Drive for training
- Saves trained LoRA models to Google Drive
- Uses Google Service Account for authentication

## Nodes

### Load Caption Image Pair From Google Drive

Loads batches of images and their corresponding text captions from a Google Drive folder for LoRA training.

**Inputs:**
- `folder_id`: Google Drive folder ID (the long string from the folder URL)
- `credentials_json`: Google Service Account credentials as JSON string
- `clip`: CLIP model for encoding captions

**Output:**
- Batched images and encoded conditioning for training
- Automatically pairs images with their corresponding `.txt` caption files

### Save Lora To Google Drive

Saves trained LoRA models directly to Google Drive with optional step tracking in filenames.

**Inputs:**
- `lora`: The trained LoRA model to save
- `prefix`: Filename prefix for the saved model (default: "Comfy-trained-loras")
- `folder_id`: Google Drive folder ID where the LoRA will be saved
- `credentials_json`: Google Service Account credentials as JSON string
- `steps` (optional): Training step count for filename tracking

**Output:** Saves `.safetensors` file to Google Drive

## Installation

Clone into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/yourusername/comfyui-cheaptrainerutils.git
pip install google-api-python-client google-auth safetensors
```

Restart ComfyUI.

## Setup

### Getting Google Service Account Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Drive API for your project
4. Navigate to "IAM & Admin" > "Service Accounts"
5. Create a new service account with Drive permissions
6. Generate a JSON key for the service account
7. Copy the entire JSON content to use in the node's `credentials_json` input

## Usage

### Loading Training Data

1. Upload your training images and caption files to a Google Drive folder
   - Each image should have a corresponding `.txt` file with the same name
   - Example: `image1.png` and `image1.txt`
2. Get the folder ID from the Google Drive URL
3. Add the "Load Caption Image Pair From Google Drive" node to your workflow
4. Paste your credentials JSON and folder ID
5. Connect the outputs to your training node

### Saving Trained LoRAs

1. Get a Google Service Account JSON (see Setup section)
2. Add the "Save Lora To Google Drive" node to your workflow
3. Paste your credentials JSON and the target folder ID
4. Connect your trained LoRA output
5. Run your workflow

The nodes validate folder access before uploading and provide clear error messages if something goes wrong.

## Contributing

If you fix something, PR it.

## License

MIT License

## Why This Exists

Made this for training LoRAs on cheap vast.ai instances. When the instance dies, you lose everything. This pushes your trained models to Drive automatically. That's it.
