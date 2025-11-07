import folder_paths
import json

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
                        "tooltip": "Google Service Account credentials JSON content"
                        }
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
    FUNCTION = "save"
    CATEGORY = "loaders"
    EXPERIMENTAL = True
    OUTPUT_NODE = True

    def save(self, lora, prefix, credentials_json steps=None):
        from googleapiclient.discovery import build
        from google.oauth2 import service_account

        credentials_dict = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(
                credentials_dict,
                scopes=['https://www.googleapis.com/auth/drive.file']
            )
        drive_service
        if steps is None:
            output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        else:
            output_checkpoint = f"{filename}_{steps}_steps_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)
        safetensors.torch.save_file(lora, output_checkpoint)
        return {}


class textImagePairFromGoogleDrive:
    def __init__():
        return
    @classmethod
    def INPUT_TYPES(s):
                return {
                    "required": {
                        "images": ("IMAGE", {"tooltip": "The imges to save."}),
                        "caption": ("STRING", {"multiline": True, "default": ""})
                        "credentials_json": (
                            "STRING",
                            {
                            "multiline": True,
                            "default": "",
                            "tooltip": "Google Service Account credentials JSON content"
                            }
                        ),
                    }
    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "loaders"
    EXPERIMENTAL = True
    OUTPUT_NODE = True
