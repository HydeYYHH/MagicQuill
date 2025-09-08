import os
import gradio as gr
import random
import torch
import numpy as np
from PIL import Image, ImageOps
from MagicQuill import folder_paths
from MagicQuill.controlnet_brushnet import ControlNetBrushNetModel
from MagicQuill.grounded_segment_anything import GroundedSegmentAnything
from MagicQuill.cloth_segment import ClothSegment
import time
import argparse
import os

parser = argparse.ArgumentParser(description='MagicQuill Application')
parser.add_argument('--cloth-segment', action='store_true', help='Use cloth segmentation algorithm')
parser.add_argument('--sam-segment', action='store_true', help='Use SAM segmentation algorithm')
args = parser.parse_args()

USE_CLOTH_SEGMENT = args.cloth_segment
USE_SAM_SEGMENT = args.sam_segment

if not USE_CLOTH_SEGMENT and not USE_SAM_SEGMENT:
    USE_SAM_SEGMENT = True

AUTO_SAVE = False
RES = 512
ORIGINAL_IMAGE_PIL = None
MASK_IMAGE = None

# --- Model Initialization ---
controlNetBrushNetModel = ControlNetBrushNetModel()
groundedSegmentAnything = GroundedSegmentAnything() if (USE_SAM_SEGMENT or not USE_CLOTH_SEGMENT) else None
clothSegment = ClothSegment() if USE_CLOTH_SEGMENT else None

print(f"Using {'Cloth Segmentation' if USE_CLOTH_SEGMENT else 'SAM Segment Anything'} algorithm")

# --- Utility Functions ---

def tensor_to_pil_image(tensor):
    """Converts a Tensor to a PIL image."""
    tensor = tensor.squeeze(0) * 255.
    return Image.fromarray(tensor.cpu().byte().numpy())

def load_and_resize_tensor(pil_image, max_size=512):
    """Converts a PIL image to a resized Tensor required by the model."""
    image = pil_image.convert('RGB')
    width, height = image.size
    scaling_factor = max_size / min(width, height)
    new_size = (int(width * scaling_factor), int(height * scaling_factor))
    image = image.resize(new_size, Image.LANCZOS)
    image_array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_array)[None,]

# --- Gradio Event Handlers ---

def on_upload(image_path):
    """
    Handles a new image upload. This is the ONLY place where the original image state is set.
    It resets everything to ensure a clean start.
    """
    global ORIGINAL_IMAGE_PIL, MASK_IMAGE
    if image_path is None:
        ORIGINAL_IMAGE_PIL = None
        MASK_IMAGE = None
        return None  # Clear the mask preview

    # Read the image and store it in the global variable
    ORIGINAL_IMAGE_PIL = Image.open(image_path)
    ORIGINAL_IMAGE_PIL = ImageOps.exif_transpose(ORIGINAL_IMAGE_PIL)
    MASK_IMAGE = None  # Reset any previous mask

    # Return None to clear the mask preview area
    return gr.update(value=None)

def generate_mask_handler(text_prompt, selected_clothes=None):
    """
    Generates a mask using the globally stored original image.
    Outputs the annotated preview without touching the original image component.
    """
    global MASK_IMAGE

    if ORIGINAL_IMAGE_PIL is None:
        raise gr.Error("Please upload an original image first.")
    
    if not USE_CLOTH_SEGMENT and (not text_prompt or not text_prompt.strip()):
        raise gr.Error("Text prompt for mask generation cannot be empty.")

    if USE_CLOTH_SEGMENT:
        img_copy = ORIGINAL_IMAGE_PIL.copy()
        
        clothes_to_segment = selected_clothes if selected_clothes else ["Upper-clothes", "Skirt", "Pants", "Dress"]
        
        mask_pil, annotated_pil = clothSegment.generate_mask(img_copy, clothes=clothes_to_segment)
    else:
        mask_pil, annotated_pil = groundedSegmentAnything.generate_mask(ORIGINAL_IMAGE_PIL, text_prompt)

    # Store the generated black-and-white mask in our global state
    MASK_IMAGE = mask_pil

    # Return the annotated image to be displayed in the dedicated preview component
    return annotated_pil

def generate_image_handler(
    positive_prompt, ckpt_name, negative_prompt, grow_size, edge_strength,
    pose_strength, depth_strength, inpaint_strength, seed, steps, cfg,
    sampler_name, scheduler
):
    """
    Generates the final image using the globally stored original image and mask.
    This function is now independent of the UI component values.
    """
    if ORIGINAL_IMAGE_PIL is None:
        raise gr.Error("An original image must be uploaded first.")
    if MASK_IMAGE is None:
        raise gr.Error("A mask must be generated first by clicking 'Generate Mask'.")
    if not positive_prompt or not positive_prompt.strip():
        raise gr.Error("Positive prompt cannot be empty.")
    if not ckpt_name:
        raise gr.Error("A base model must be selected.")

    # Prepare Tensors on-the-fly from the reliable global state
    image_tensor = load_and_resize_tensor(ORIGINAL_IMAGE_PIL, max_size=RES)
    mask_array = np.array(MASK_IMAGE.convert('L')).astype(np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_array)[None,]

    # Ensure the mask and image Tensor dimensions are compatible
    if mask_tensor.shape[1:] != image_tensor.shape[1:3]:
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0).float(),
            size=(image_tensor.shape[1], image_tensor.shape[2]),
            mode='nearest'
        ).squeeze(0)

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    with torch.no_grad():
        result = controlNetBrushNetModel.process(
            ckpt_name, image_tensor, positive_prompt, negative_prompt,
            mask_tensor, grow_size, edge_strength, pose_strength,
            depth_strength, inpaint_strength, seed, int(steps), cfg,
            sampler_name, scheduler, base_model_version='SD1.5', dtype='float16'
        )

    final_pil_image = tensor_to_pil_image(result[1])

    if AUTO_SAVE:
        auto_save_generated_image(final_pil_image)

    # Clean up memory
    del mask_tensor, result, image_tensor
    torch.cuda.empty_cache()

    return final_pil_image

def auto_save_generated_image(pil_image):
    """Saves the generated image to the 'output' folder."""
    os.makedirs("output", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join("output", f"magicquill_{timestamp}.png")
    pil_image.save(save_path)
    print(f"Image saved to: {save_path}")

# --- Gradio UI Layout ---
css = ".row { width: 90%; margin: auto; } footer { visibility: hidden }"

CLOTHES_OPTIONS = [
    "Hat", "Upper-clothes", "Skirt", "Pants", 
    "Dress", "Belt", "Left-shoe", "Right-shoe", "Scarf"
]

with gr.Blocks(css=css) as demo:
    with gr.Row(elem_classes="row"):
        with gr.Column():
            original_image = gr.Image(type="filepath", label="1. Upload Original Image")
            # New, dedicated mask preview area
            mask_preview_image = gr.Image(type="pil", label="Mask Preview", interactive=False)
            if USE_CLOTH_SEGMENT:
                selected_clothes = gr.CheckboxGroup(
                    label="2. Select Clothes Types",
                    choices=CLOTHES_OPTIONS,
                    value=["Upper-clothes", "Skirt", "Pants", "Dress"]
                )
                text_prompt = gr.Textbox(label="Text Prompt (for SAM algorithm)", visible=False)
            else:
                text_prompt = gr.Textbox(label="2. Enter Text for Mask Generation", value="shirt")
                selected_clothes = gr.CheckboxGroup(choices=CLOTHES_OPTIONS, visible=False)
            mask_btn = gr.Button("Generate Mask", variant="secondary")
            positive_prompt = gr.Textbox(label="3. Enter Positive Prompt", value="")

        with gr.Column():
            output_image = gr.Image(type="pil", label="Generated Image")
            btn = gr.Button("4. Run", variant="primary")

    with gr.Accordion("Advanced Parameters", open=False):
        # Safely get the list of checkpoints
        ckpt_list = folder_paths.get_filename_list("checkpoints")

        with gr.Row():
            ckpt_name = gr.Dropdown(
                label="Base Model Name",
                choices=ckpt_list,
                value=ckpt_list[0] if ckpt_list else None,  # Safe initialization
                interactive=True
            )
            auto_save_checkbox = gr.Checkbox(label="Auto Save", value=False, interactive=True)
            resolution_slider = gr.Slider(
                label="Resolution", minimum=256, maximum=2048, value=512, step=64, interactive=True
            )
        with gr.Row():
            negative_prompt = gr.Textbox(label="Negative Prompt", value="worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch", interactive=True, lines=2)
            seed = gr.Number(label="Seed", value=-1, precision=0, interactive=True)
            steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=20, step=1, interactive=True)
            cfg = gr.Slider(label="CFG", minimum=0.0, maximum=20.0, value=5.0, step=0.1, interactive=True)
        with gr.Row():
             grow_size = gr.Slider(label="Mask Grow Size", minimum=0, maximum=100, value=15, step=1, interactive=True)
             edge_strength = gr.Slider(label="Edge Strength", minimum=0.0, maximum=5.0, value=0.55, step=0.01, interactive=True)
             pose_strength = gr.Slider(label="Pose Strength", minimum=0.0, maximum=5.0, value=0.55, step=0.01, interactive=True)
             depth_strength = gr.Slider(label="Depth Strength", minimum=0.0, maximum=5.0, value=0.55, step=0.01, interactive=True)
             inpaint_strength = gr.Slider(label="Inpaint Strength", minimum=0.0, maximum=5.0, value=1.0, step=0.01, interactive=True)
        with gr.Row():
             sampler_name = gr.Dropdown(label="Sampler Name", choices=["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms", "dpmpp_2s_ancestral", "dpmpp_2m", "dpmpp_sde", "ddim", "uni_pc"], value='dpmpp_2m', interactive=True)
             scheduler = gr.Dropdown(label="Scheduler", choices=["normal", "karras", "exponential", "simple"], value='karras', interactive=True)

    # --- Callback Function Definitions ---
    def update_auto_save(value): global AUTO_SAVE; AUTO_SAVE = value
    def update_resolution(value): global RES; RES = value

    # --- Event Listener Wiring ---
    auto_save_checkbox.change(fn=update_auto_save, inputs=auto_save_checkbox)
    resolution_slider.change(fn=update_resolution, inputs=resolution_slider)

    # When an image is uploaded or cleared, call the handler to reset the state
    original_image.upload(on_upload, inputs=[original_image], outputs=[mask_preview_image])
    original_image.clear(on_upload, inputs=[original_image], outputs=[mask_preview_image])

    # The mask button uses the appropriate inputs based on the selected algorithm
    mask_btn.click(
        generate_mask_handler,
        inputs=[text_prompt, selected_clothes],
        outputs=[mask_preview_image]
    )

    # The run button receives all parameters (but not images, as it uses the reliable global state)
    btn.click(
        generate_image_handler,
        inputs=[
            positive_prompt, ckpt_name, negative_prompt, grow_size, edge_strength,
            pose_strength, depth_strength, inpaint_strength, seed, steps, cfg,
            sampler_name, scheduler
        ],
        outputs=[output_image]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_port=7860)