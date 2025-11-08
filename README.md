# ComfyUI Cheap Trainer Utils

ComfyUI nodes for managing LoRA training workflows with Google Drive. Built for temporary cloud instances where you need to save training outputs before the instance shuts down.

## What It Does

- Loads image-caption pairs from Google Drive for training
- Saves trained LoRA models to Google Drive
- Loads trained LoRAs back from Google Drive
- Caches training data locally to avoid re-downloading
- Uses Google Service Account or OAuth for authentication

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

### Load Caption Image Pair From Google Drive (Cached)

Same as the regular loader but caches downloaded images locally. On first run, downloads everything from Google Drive and saves to `ComfyUI/input/gdrive_cache/{folder_id}/`. Subsequent runs load from cache instead of re-downloading.

**Inputs:**
- `folder_id`: Google Drive folder ID
- `credentials_json`: Google Service Account credentials as JSON string
- `clip`: CLIP model for encoding captions

**Output:**
- Batched images and encoded conditioning from cache or fresh download

### Save Lora To Google Drive

Saves trained LoRA models directly to Google Drive with optional step tracking in filenames. Uses OAuth authentication.

**Inputs:**
- `lora`: The trained LoRA model to save
- `prefix`: Filename prefix for the saved model (default: "Comfy-trained-loras")
- `client_id`: OAuth Client ID from Google Cloud Console
- `client_secret`: OAuth Client Secret from Google Cloud Console
- `refresh_token`: OAuth Refresh Token (generate once using `oauth_setup.py`)
- `folder_id`: Google Drive folder ID where the LoRA will be saved
- `steps` (optional): Training step count for filename tracking

**Output:** Saves `.safetensors` file to Google Drive

### Load Lora From Google Drive

Loads a trained LoRA from Google Drive and applies it to your model and CLIP. Downloads fresh every time (no caching).

**Inputs:**
- `model`: The diffusion model the LoRA will be applied to
- `clip`: The CLIP model the LoRA will be applied to
- `folder_id`: Google Drive folder ID containing the LoRA
- `lora_name`: Name of the LoRA file (e.g., "my_lora.safetensors")
- `credentials_json`: Google Service Account credentials as JSON string
- `strength_model`: How strongly to modify the diffusion model (-100.0 to 100.0, default: 1.0)
- `strength_clip`: How strongly to modify the CLIP model (-100.0 to 100.0, default: 1.0)

**Output:** Modified model and CLIP with LoRA applied

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

**Pro tip:** Use the cached version if you're running multiple training sessions with the same dataset. First run downloads everything, subsequent runs are instant.

### Saving Trained LoRAs

1. Set up OAuth credentials (run `oauth_setup.py` to get your refresh token)
2. Add the "Save Lora To Google Drive" node to your workflow
3. Paste your client ID, client secret, refresh token, and target folder ID
4. Connect your trained LoRA output
5. Run your workflow

### Loading LoRAs from Google Drive

1. Add the "Load Lora From Google Drive" node to your workflow
2. Connect your model and CLIP inputs
3. Paste your credentials JSON and folder ID
4. Enter the exact filename of the LoRA (e.g., "my_lora_500_steps.safetensors")
5. Adjust strength sliders as needed
6. Connect outputs to your generation workflow

The nodes validate folder access before uploading and provide clear error messages if something goes wrong.

## Contributing

If you fix something, PR it.

## License

MIT License

## Why This Exists

Made this for training LoRAs on cheap vast.ai instances. When the instance dies, you lose everything. This pushes your trained models to Drive automatically. That's it.
