import folder_paths
import json
import os
import logging
import safetensors.torch
from io import BytesIO
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account
from PIL import Image
import numpy as np
import torch
from comfy.comfy_types.node_typing import IO


class SaveLoratoGoogleDrive:
    def __init__(self):
        self.gdrive_saved_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora": (
                    IO.LORA_MODEL,
                    {
                        "tooltip": "The LoRA model to save. Do not use the model with LoRA layers."
                    },
                ),
                "prefix": (
                    "STRING",
                    {
                        "default": "Comfy-trained-loras",
                        "tooltip": "The prefix to use for the saved LoRA file.",
                    },
                ),
                "credentials_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Google Service Account credentials JSON content",
                    },
                ),
                "folder_id": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Google Drive folder ID",
                    },
                ),
            },
            "optional": {
                "steps": (
                    IO.INT,
                    {
                        "forceInput": True,
                        "tooltip": "Optional: The number of steps to LoRA has been trained for, used to name the saved file.",
                    },
                ),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "googledrivelorasave"
    CATEGORY = "cheap_trainer_utils"
    EXPERIMENTAL = True
    OUTPUT_NODE = True

    def googledrivelorasave(
        self, lora, prefix, folder_id, credentials_json, steps=None
    ):
        # Validate folder_id
        if not folder_id or len(folder_id.strip()) < 10:
            raise ValueError(
                "folder_id must be a valid Google Drive folder ID (the long string of letters/numbers), "
                "not the folder name like 'My LoRAs'."
            )

        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(prefix, self.gdrive_saved_dir)
        )
        credentials_dict = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict, scopes=["https://www.googleapis.com/auth/drive"]
        )
        drive_service = build("drive", "v3", credentials=credentials, cache_discovery=False)

        # Verify folder exists and is accessible
        try:
            folder_info = drive_service.files().get(
                fileId=folder_id,
                fields='id,name,mimeType'
            ).execute()
            logging.info(f"✅ Can access folder: {folder_info.get('name')} (ID: {folder_info['id']})")
        except Exception as e:
            raise ValueError(f"Cannot access folder {folder_id}: {e}")

        if steps is None:
            output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        else:
            output_checkpoint = f"{filename}_{steps}_steps_{counter:05}_.safetensors"

        local_path = os.path.join(full_output_folder, output_checkpoint)
        file_metadata = {"name": output_checkpoint, "parents": [folder_id]}

        # Save LoRA locally first
        safetensors.torch.save_file(lora, local_path)

        # Upload to Google Drive
        media = MediaFileUpload(local_path, mimetype="application/octet-stream")
        file = (
            drive_service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )

        return {}


class textImagePairFromGoogleDrive:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_id": ("STRING", {"default": "", "tooltip": "Google Drive folder ID"}),
                "credentials_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Google Service Account credentials JSON content",
                    },
                ),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
            },
        }

    RETURN_TYPES = ("IMAGE", IO.CONDITIONING)
    FUNCTION = "textImagePairing"
    CATEGORY = "cheap_trainer_utils"
    EXPERIMENTAL = True
    DESCRIPTION = "Loads a batch of images and captions from Google Drive for training."

    def textImagePairing(self, folder_id, clip, credentials_json):
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

        logging.info(f"Loading images from Google Drive folder: {folder_id}")

        credentials_dict = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict, scopes=["https://www.googleapis.com/auth/drive"]
        )

        drive_service = build("drive", "v3", credentials=credentials, cache_discovery=False)

        # Test if we can access the folder itself
        try:
            folder_info = drive_service.files().get(
                fileId=folder_id,
                fields='id,name,mimeType,capabilities'
            ).execute()
            logging.info(f"✅ Can access folder: {folder_info.get('name')} (ID: {folder_info['id']})")
            logging.info(f"Folder capabilities: {folder_info.get('capabilities')}")
        except Exception as e:
            logging.error(f"❌ Cannot access folder: {e}")
            raise ValueError(f"Cannot access folder {folder_id}: {e}")

        # Query to get all files in the specified folder
        query = f"'{folder_id}' in parents and trashed=false"
        logging.info(f"Query: {query}")
        results = (
            drive_service.files()
            .list(
                q=query,
                corpora='user',
                includeItemsFromAllDrives=False,
                supportsAllDrives=False,
                fields="files(id, name, mimeType)",
                pageSize=1000
            )
            .execute()
        )
        
        logging.info(f"Found {len(results['files'])} files in the folder.")
        
        files = results.get('files', [])
        valid_extensions = [".png", ".jpg", ".jpeg", ".webp"]
        images = {
            f["name"].rsplit(".", 1)[0]: f
            for f in files
            if any(f["name"].lower().endswith(ext) for ext in valid_extensions)
        }
        captions = {
            f["name"].rsplit(".", 1)[0]: f
            for f in files
            if f["name"].lower().endswith(".txt")
        }
        logging.info(f"Found {len(images)} images and {len(captions)} captions.")
        
        paired_names = sorted(set(images.keys()) & set(captions.keys()))

        if not paired_names:
            raise ValueError("No matching image-caption pairs found in the folder!")

        # Load all images and captions
        output_images = []
        caption_texts = []

        for pair_name in paired_names:
            image_file = images[pair_name]
            caption_file = captions[pair_name]

            # Download image
            image_request = drive_service.files().get_media(fileId=image_file["id"])
            image_buffer = BytesIO()
            image_downloader = MediaIoBaseDownload(image_buffer, image_request)
            done = False
            while not done:
                status, done = image_downloader.next_chunk()

            # Download caption
            caption_request = drive_service.files().get_media(fileId=caption_file["id"])
            caption_buffer = BytesIO()
            caption_downloader = MediaIoBaseDownload(caption_buffer, caption_request)
            done = False
            while not done:
                status, done = caption_downloader.next_chunk()

            # Process image
            image_buffer.seek(0)
            pil_image = Image.open(image_buffer)
            pil_image = pil_image.convert("RGB")
            image_np = np.array(pil_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            output_images.append(image_tensor)

            # Read caption text
            caption_buffer.seek(0)
            caption_text = caption_buffer.read().decode("utf-8").strip()
            caption_texts.append(caption_text)

        # Batch all images
        output_tensor = torch.cat(output_images, dim=0)

        logging.info(f"Loaded {len(output_tensor)} images from Google Drive.")

        # Encode all captions with CLIP
        logging.info(f"Encoding captions from Google Drive.")
        conditions = []
        empty_cond = clip.encode_from_tokens_scheduled(clip.tokenize(""))
        for text in caption_texts:
            if text == "":
                conditions.append(empty_cond)
            else:
                tokens = clip.tokenize(text)
                conditions.extend(clip.encode_from_tokens_scheduled(tokens))

        logging.info(f"Encoded {len(conditions)} captions from Google Drive.")
        return (output_tensor, conditions)


NODE_CLASS_MAPPINGS = {
    "Save Lora To Google Drive": SaveLoratoGoogleDrive,
    "Load Caption Image Pair From Google Drive": textImagePairFromGoogleDrive,
}
