# ComfyUI Cheap Trainer Utils

**⚠️ v0.0.1 - Early Development Version**

Basic ComfyUI nodes for uploading LoRA models and training data to Google Drive. Built for temporary cloud instances where you need to save training outputs before the instance shuts down.

**Currently only tested on vast.ai instances.** May or may not work elsewhere.

## What It Does

- Saves trained LoRA models to Google Drive
- Saves image-caption pairs to Google Drive
- Uses Google Service Account for auth

## Nodes

### SaveLoratoGoogleDrive

Saves trained LoRA models directly to Google Drive with optional step tracking in filenames.

**Inputs:**
- `lora`: The trained LoRA model to save
- `prefix`: Filename prefix for the saved model (default: "Comfy-trained-loras")
- `credentials_json`: Google Service Account credentials as JSON string
- `steps` (optional): Training step count for filename tracking

**Output:** Saves `.safetensors` file to Google Drive

### textImagePairFromGoogleDrive

Saves image and caption pairs to Google Drive for training dataset management.

**Inputs:**
- `images`: Images to save
- `caption`: Text caption/description for the images
- `credentials_json`: Google Service Account credentials as JSON string

**Output:** Uploads image-caption pairs to Google Drive

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

1. Get a Google Service Account JSON (see Setup section)
2. Add the node to your workflow
3. Paste your credentials JSON
4. Connect your LoRA output
5. Run it

That's it. If it works, your LoRA goes to Drive. If it doesn't, good luck debugging.

## Known Issues

- Barely tested
- Code is incomplete in places
- No error handling to speak of
- Only tested on vast.ai, YMMV elsewhere

## Contributing

If you fix something, PR it.

## License

MIT License

## Why This Exists

Made this for training LoRAs on cheap vast.ai instances. When the instance dies, you lose everything. This pushes your trained models to Drive automatically. That's it.
