import folder_paths
import json
import os
import logging
import safetensors.torch
from io import BytesIO
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from PIL import Image
import numpy as np
import torch
import tempfile
import comfy.utils
import comfy.sd
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
                "client_id": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "OAuth Client ID from Google Cloud Console",
                    },
                ),
                "client_secret": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "OAuth Client Secret from Google Cloud Console",
                    },
                ),
                "refresh_token": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "OAuth Refresh Token (generate once using oauth_setup.py)",
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
        self, lora, prefix, client_id, client_secret, refresh_token, folder_id, steps=None
    ):
        if not folder_id or len(folder_id.strip()) < 10:
            raise ValueError(
                "folder_id must be a valid Google Drive folder ID (the long string of letters/numbers), "
                "not the folder name like 'My LoRAs'."
            )

        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(prefix, self.gdrive_saved_dir)
        )

        credentials = Credentials(
            token=None,
            refresh_token=refresh_token.strip(),
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id.strip(),
            client_secret=client_secret.strip(),
            scopes=['https://www.googleapis.com/auth/drive']
        )

        credentials.refresh(Request())

        drive_service = build("drive", "v3", credentials=credentials, cache_discovery=False)

        try:
            folder_info = drive_service.files().get(
                fileId=folder_id,
                fields='id,name,mimeType'
            ).execute()
        except Exception as e:
            raise ValueError(f"Cannot access folder {folder_id}: {e}")

        if steps is None:
            output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        else:
            output_checkpoint = f"{filename}_{steps}_steps_{counter:05}_.safetensors"

        local_path = os.path.join(full_output_folder, output_checkpoint)
        file_metadata = {"name": output_checkpoint, "parents": [folder_id]}

        safetensors.torch.save_file(lora, local_path)

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

        credentials_dict = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict, scopes=["https://www.googleapis.com/auth/drive"]
        )

        drive_service = build("drive", "v3", credentials=credentials, cache_discovery=False)

        try:
            folder_info = drive_service.files().get(
                fileId=folder_id,
                fields='id,name,mimeType,capabilities'
            ).execute()
        except Exception as e:
            raise ValueError(f"Cannot access folder {folder_id}: {e}")

        query = f"'{folder_id}' in parents and trashed=false"
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

        paired_names = sorted(set(images.keys()) & set(captions.keys()))

        if not paired_names:
            raise ValueError("No matching image-caption pairs found in the folder!")

        output_images = []
        caption_texts = []

        for pair_name in paired_names:
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
            output_images.append(image_tensor)

            caption_buffer.seek(0)
            caption_text = caption_buffer.read().decode("utf-8").strip()
            caption_texts.append(caption_text)

        output_tensor = torch.cat(output_images, dim=0)

        conditions = []
        empty_cond = clip.encode_from_tokens_scheduled(clip.tokenize(""))
        for text in caption_texts:
            if text == "":
                conditions.append(empty_cond)
            else:
                tokens = clip.tokenize(text)
                conditions.extend(clip.encode_from_tokens_scheduled(tokens))

        return (output_tensor, conditions)


class textImagePairFromGoogleDriveCached:
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
    DESCRIPTION = "Loads images and captions from local cache first. If cache doesn't exist, downloads from Google Drive and saves to cache."

    def textImagePairing(self, folder_id, clip, credentials_json):
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

        cache_base_dir = os.path.join(folder_paths.get_input_directory(), "gdrive_cache")
        cache_folder = os.path.join(cache_base_dir, folder_id)

        use_cache = os.path.exists(cache_folder) and len(os.listdir(cache_folder)) > 0

        if use_cache:
            paired_names = []
            files_in_cache = os.listdir(cache_folder)

            valid_extensions = [".png", ".jpg", ".jpeg", ".webp"]
            images = {
                f.rsplit(".", 1)[0]: f
                for f in files_in_cache
                if any(f.lower().endswith(ext) for ext in valid_extensions)
            }
            captions = {
                f.rsplit(".", 1)[0]: f
                for f in files_in_cache
                if f.lower().endswith(".txt")
            }

            paired_names = sorted(set(images.keys()) & set(captions.keys()))

            if not paired_names:
                use_cache = False
            else:
                output_images = []
                caption_texts = []

                for pair_name in paired_names:
                    image_path = os.path.join(cache_folder, images[pair_name])
                    pil_image = Image.open(image_path).convert("RGB")
                    image_np = np.array(pil_image).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_np)[None,]
                    output_images.append(image_tensor)

                    caption_path = os.path.join(cache_folder, captions[pair_name])
                    with open(caption_path, 'r', encoding='utf-8') as f:
                        caption_text = f.read().strip()
                    caption_texts.append(caption_text)

                output_tensor = torch.cat(output_images, dim=0)

        if not use_cache:
            os.makedirs(cache_folder, exist_ok=True)

            credentials_dict = json.loads(credentials_json)
            credentials = service_account.Credentials.from_service_account_info(
                credentials_dict, scopes=["https://www.googleapis.com/auth/drive"]
            )

            drive_service = build("drive", "v3", credentials=credentials, cache_discovery=False)

            try:
                folder_info = drive_service.files().get(
                    fileId=folder_id,
                    fields='id,name,mimeType,capabilities'
                ).execute()
            except Exception as e:
                raise ValueError(f"Cannot access folder {folder_id}: {e}")

            query = f"'{folder_id}' in parents and trashed=false"
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

            paired_names = sorted(set(images.keys()) & set(captions.keys()))

            if not paired_names:
                raise ValueError("No matching image-caption pairs found in the folder!")

            output_images = []
            caption_texts = []

            for pair_name in paired_names:
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

                image_cache_path = os.path.join(cache_folder, image_file["name"])
                pil_image.save(image_cache_path)

                image_np = np.array(pil_image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,]
                output_images.append(image_tensor)

                caption_buffer.seek(0)
                caption_text = caption_buffer.read().decode("utf-8").strip()
                caption_texts.append(caption_text)

                caption_cache_path = os.path.join(cache_folder, caption_file["name"])
                with open(caption_cache_path, 'w', encoding='utf-8') as f:
                    f.write(caption_text)

            output_tensor = torch.cat(output_images, dim=0)

        conditions = []
        empty_cond = clip.encode_from_tokens_scheduled(clip.tokenize(""))
        for text in caption_texts:
            if text == "":
                conditions.append(empty_cond)
            else:
                tokens = clip.tokenize(text)
                conditions.extend(clip.encode_from_tokens_scheduled(tokens))

        return (output_tensor, conditions)

class loadLoraFromGoogleDrive:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "folder_id": ("STRING", {"default": "", "tooltip": "Google Drive folder ID"}),
                "lora_name": ("STRING", {"default": "", "tooltip": "Name of the LoRA file to load (e.g., 'my_lora.safetensors')"}),
                "credentials_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Google Service Account credentials JSON content",
                    },
                ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "cheap_trainer_utils"
    EXPERIMENTAL = True
    DESCRIPTION = "Loads a LoRA from Google Drive and applies it to the model and CLIP."

    def load_lora(self, model, clip, folder_id, lora_name, credentials_json, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        credentials_dict = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict, scopes=["https://www.googleapis.com/auth/drive"]
        )

        drive_service = build("drive", "v3", credentials=credentials, cache_discovery=False)

        try:
            folder_info = drive_service.files().get(
                fileId=folder_id,
                fields='id,name,mimeType'
            ).execute()
        except Exception as e:
            raise ValueError(f"Cannot access folder {folder_id}: {e}")

        query = f"'{folder_id}' in parents and name='{lora_name}' and trashed=false"
        results = (
            drive_service.files()
            .list(
                q=query,
                corpora='user',
                includeItemsFromAllDrives=False,
                supportsAllDrives=False,
                fields="files(id, name)",
                pageSize=1
            )
            .execute()
        )

        files = results.get('files', [])

        if not files:
            raise ValueError(f"LoRA file '{lora_name}' not found in Google Drive folder {folder_id}")

        lora_file = files[0]

        request = drive_service.files().get_media(fileId=lora_file["id"])
        file_buffer = BytesIO()
        downloader = MediaIoBaseDownload(file_buffer, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as temp_file:
            temp_path = temp_file.name
            file_buffer.seek(0)
            temp_file.write(file_buffer.read())

        try:
            lora = comfy.utils.load_torch_file(temp_path, safe_load=True)
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
            return (model_lora, clip_lora)
        finally:
            os.unlink(temp_path)


NODE_CLASS_MAPPINGS = {
    "Save Lora To Google Drive": SaveLoratoGoogleDrive,
    "Load Caption Image Pair From Google Drive": textImagePairFromGoogleDrive,
    "Load Caption Image Pair From Google Drive (Cached)": textImagePairFromGoogleDriveCached,
    "Load Lora From Google Drive": loadLoraFromGoogleDrive
}
