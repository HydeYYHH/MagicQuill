import os
import requests
from pathlib import Path

def download_file(url, destination):
    """Download a file from URL to destination"""
    print(f"Downloading {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded to {destination}")

def main():
    # Define base directories
    base_dir = Path(".")
    checkpoints_dir = base_dir / "MagicQuill" / "models" / "checkpoints"
    controlnet_dir = base_dir / "MagicQuill" / "models" / "controlnet"
    inpaint_dir = base_dir / "MagicQuill" / "models" / "inpaint" / "brushnet" / "random_mask_brushnet_ckpt"
    preprocessor_dir = base_dir / "MagicQuill" / "models" / "preprocessor"
    grounded_segment_anything_dir = base_dir / "MagicQuill" / "models" / "grounded_segment_anything"
    
    # Create directories
    for directory in [checkpoints_dir, controlnet_dir, inpaint_dir, preprocessor_dir, grounded_segment_anything_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Download models
    models = [
        # Checkpoint model
        ("https://huggingface.co/LiuZichen/MagicQuill-models/resolve/main/checkpoints/SD1.5/realisticVisionV60B1_v51VAE.safetensors?download=true",
         checkpoints_dir / "realisticVisionV60B1_v51VAE.safetensors"),
        
        # ControlNet models
        ("https://huggingface.co/LiuZichen/MagicQuill-models/resolve/main/controlnet/control_v11p_sd15_scribble.safetensors?download=true",
         controlnet_dir / "control_v11p_sd15_scribble.safetensors"),
        ("https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/diffusion_pytorch_model.safetensors?download=true",
         controlnet_dir / "control_v11p_sd15_openpose.safetensors"),
        ("https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth/resolve/main/diffusion_pytorch_model.safetensors?download=true",
         controlnet_dir / "control_v11f1p_sd15_depth.safetensors"),
        
        # Inpaint model
        ("https://huggingface.co/LiuZichen/MagicQuill-models/resolve/main/inpaint/brushnet/random_mask_brushnet_ckpt/diffusion_pytorch_model.safetensors?download=true",
         inpaint_dir / "diffusion_pytorch_model.safetensors"),
        ("https://huggingface.co/LiuZichen/MagicQuill-models/resolve/main/inpaint/brushnet/random_mask_brushnet_ckpt/config.json?download=true",
         inpaint_dir / "config.json"),
         
        # Preprocessor models
        ("https://huggingface.co/LiuZichen/MagicQuill-models/resolve/main/preprocessor/table5_pidinet.pth?download=true",
         preprocessor_dir / "table5_pidinet.pth"),
        ("https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth",
         preprocessor_dir / "lightweight_openpose.pth"),
        
        # Grounded Segment Anything models
        ("https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
         grounded_segment_anything_dir / "groundingdino_swint_ogc.pth"),
        ("https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/sams/sam_vit_b_01ec64.pth?download=true",
         grounded_segment_anything_dir / "sam_vit_b_01ec64.pth"),
    ]
    
    # Download all models
    for url, destination in models:
        download_file(url, destination)
    
    print("All models downloaded successfully!")

if __name__ == "__main__":
    main()