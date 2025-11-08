import folder_paths
import json
import os
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
                        "default": "My LoRAs",
                        "tooltip": "Folder ID to save the LoRA file. LAST LONG STRING OF NUMBERS AND LETTERS",
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
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(prefix, self.gdrive_saved_dir)
        )
        credentials_dict = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict, scopes=["https://www.googleapis.com/auth/drive.file"]
        )
        drive_service = build("drive", "v3", credentials=credentials)

        if steps is None:
            output_checkpoint = f"{prefix}_.safetensors"
        else:
            output_checkpoint = f"{prefix}_{steps}_steps_.safetensors"
        file_metadata = {"name": output_checkpoint, "parents": [folder_id]}
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)
        safetensors.torch.save_file(lora, output_checkpoint)
        media = MediaFileUpload(output_checkpoint, mimetype="application/octet-stream")
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
                "folder_path": ("STRING", {"multiline": True, "default": ""}),
                "credentials_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Google Service Account credentials JSON content",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "CONDITIONING")
    FUNCTION = "textImagePairing"
    CATEGORY = "cheap_trainer_utils"
    EXPERIMENTAL = True
    OUTPUT_NODE = True

    def textImagePairing(self, folder_path, credentials_json):
        credentials_dict = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict, scopes=["https://www.googleapis.com/auth/drive.file"]
        )

        drive_service = build("drive", "v3", credentials=credentials)

        results = (
            drive_service.files()
            .list(q=query, fields="files(id, name, mimeType)", pageSize=1000)
            .execute()
        )
        images = {
            f["name"].rsplit(".", 1)[0]: f
            for f in files
            if f["name"].lower().endswith(".png")
        }
        captions = {
            f["name"].rsplit(".", 1)[0]: f
            for f in files
            if f["name"].lower().endswith(".txt")
        }

        paired_names = sorted(set(images.keys()) & set(captions.keys()))

        if not paired_names:
            raise ValueError("No matching image-caption pairs found in the folder!")
        if index >= len(paired_names):
            raise ValueError(
                f"Index {index} out of range. Found {len(paired_names)} pairs."
            )
        pair_name = paired_names[index]
        image_file = images[pair_name]
        caption_file = captions[pair_name]

        image_request = drive_service.files().get_media(fileId=image_file["id"])
        image_buffer = BytesIO()
        image_downloader = MediaIoBaseDownload(image_buffer, image_request)
        done = False
        while not done:
            status, done = image_downloader.next_chunk()

        caption_request = drive_service.files().get_media(fileId=caption_file["id"])
        caption_buffer = BytesIO()
        caption_downloader = MediaIoBaseDownload(caption_buffer, caption_request)
        done = False
        while not done:
            status, done = caption_downloader.next_chunk()

        image_buffer.seek(0)
        pil_image = Image.open(image_buffer)
        pil_image = pil_image.convert("RGB")

        image_np = np.array(pil_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        caption_buffer.seek(0)
        caption_text = caption_buffer.read().decode("utf-8").strip()

        return (image_tensor, caption_text)


NODE_CLASS_MAPPINGS = {
    "Save Lora To Google Drive": SaveLoratoGoogleDrive,
    "Load Caption Image Pair From Google Drive": textImagePairFromGoogleDrive,
}
