import os
import gradio as gr
import random
import torch
import numpy as np
from PIL import Image, ImageOps
from MagicQuill import folder_paths
from MagicQuill.controlnet_brushnet import ControlNetBrushNetModel
from MagicQuill.grounded_segment_anything import GroundedSegmentAnything
import time

AUTO_SAVE = False
RES = 512
ANNOTATED_IMAGE = None
MASK_IMAGE = None
ORIGINAL_IMAGE_TENSOR = None

controlNetBrushNetModel = ControlNetBrushNetModel()
groundedSegmentAnything = GroundedSegmentAnything()

def tensor_to_pil_image(tensor):
    tensor = tensor.squeeze(0) * 255.
    pil_image = Image.fromarray(tensor.cpu().byte().numpy())
    return pil_image

def read_image_from_path(image_path):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    return image

def create_alpha_mask(image_path):
    """Create an alpha mask from the alpha channel of an image."""
    image = read_image_from_path(image_path)
    mask = torch.zeros((1, image.height, image.width), dtype=torch.float32, device="cpu")
    if 'A' in image.getbands():
        alpha_channel = np.array(image.getchannel('A')).astype(np.float32) / 255.0
        mask[0] = 1.0 - torch.from_numpy(alpha_channel)
    return mask

def load_and_preprocess_image(image_path, convert_to='RGB'):
    """Load and preprocess an image from a file path."""
    image = read_image_from_path(image_path)
    image = image.convert(convert_to)
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array)[None,]
    return image_tensor

def load_and_resize_image(image_path, convert_to='RGB', max_size=512):
    """Load and preprocess an image from a file path, resize if necessary."""
    image = read_image_from_path(image_path)
    image = image.convert(convert_to)
    width, height = image.size
    scaling_factor = max_size / min(width, height)
    new_size = (int(width * scaling_factor), int(height * scaling_factor))
    image = image.resize(new_size, Image.LANCZOS)
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array)[None,]
    return image_tensor

def prepare_masks():
    # Use the global annotated image and mask
    global ANNOTATED_IMAGE, MASK_IMAGE, ORIGINAL_IMAGE_TENSOR
    if ANNOTATED_IMAGE is None or MASK_IMAGE is None:
        raise ValueError("Mask and annotated image must be generated first by 'Generate Mask' button.")
    # Convert mask to tensor
    mask_array = np.array(MASK_IMAGE).astype(np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_array)[None,]
    
    if ORIGINAL_IMAGE_TENSOR is not None and mask_tensor.shape[2:] != ORIGINAL_IMAGE_TENSOR.shape[1:3]:
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(1).float(), 
            size=(ORIGINAL_IMAGE_TENSOR.shape[1], ORIGINAL_IMAGE_TENSOR.shape[2]), 
            mode='nearest'
        ).squeeze(1).round()
    
    return mask_tensor


def generate(ckpt_name, positive_prompt, negative_prompt, grow_size, edge_strength, pose_strength, depth_strength, inpaint_strength, seed, steps, cfg, sampler_name, scheduler, progress=None):
    global ORIGINAL_IMAGE_TENSOR
    mask_tensor = prepare_masks()
    
    with torch.no_grad():
        result = controlNetBrushNetModel.process(
            ckpt_name,
            ORIGINAL_IMAGE_TENSOR, 
            positive_prompt, 
            negative_prompt, 
            mask_tensor, 
            grow_size, 
            edge_strength, 
            pose_strength,
            depth_strength,
            inpaint_strength, 
            seed, 
            steps, 
            cfg, 
            sampler_name, 
            scheduler,
            base_model_version='SD1.5',
            dtype='float16'
        )

    final_pil_image = tensor_to_pil_image(result[1])
    
    # Free up memory by deleting temporary variables
    del mask_tensor, result
    
    # Force garbage collection and empty CUDA cache
    torch.cuda.empty_cache()
    
    return final_pil_image

def generate_mask_handler(original_image_path, text_prompt):
    global ANNOTATED_IMAGE, MASK_IMAGE, ORIGINAL_IMAGE_TENSOR
    
    # Validate required inputs
    if not original_image_path:
        raise ValueError("Original image must be uploaded.")
    if not text_prompt or text_prompt.strip() == "":
        raise ValueError("Text prompt for mask generation cannot be empty.")
        
    # Generate mask using Grounded Segment Anything
    image_pil = read_image_from_path(original_image_path)
    mask_pil, annotated_pil = groundedSegmentAnything.generate_mask(image_pil, text_prompt)
    ANNOTATED_IMAGE = annotated_pil
    MASK_IMAGE = mask_pil
    
    ORIGINAL_IMAGE_TENSOR = load_and_resize_image(original_image_path, max_size=RES)
    return annotated_pil

def generate_image_handler(original_image_path, text_prompt, positive_prompt, ckpt_name, negative_prompt, grow_size, edge_strength, pose_strength, depth_strength, inpaint_strength, seed, steps, cfg, sampler_name, scheduler):
    global ANNOTATED_IMAGE, MASK_IMAGE, ORIGINAL_IMAGE_TENSOR
    
    # Validate required inputs
    if not original_image_path:
        raise ValueError("Original image must be uploaded.")
    if not text_prompt or text_prompt.strip() == "":
        raise ValueError("Text prompt for mask generation cannot be empty.")
    if not positive_prompt or positive_prompt.strip() == "":
        raise ValueError("Positive prompt cannot be empty.")
    if not ckpt_name:
        raise ValueError("Base model must be selected.")
    
    if ANNOTATED_IMAGE is None or MASK_IMAGE is None:
        raise ValueError("Mask and annotated image must be generated first by 'Generate Mask' button.")
    if ORIGINAL_IMAGE_TENSOR is None:
        ORIGINAL_IMAGE_TENSOR = load_and_resize_image(original_image_path, max_size=RES)
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    final_pil_image = generate(
        ckpt_name,
        positive_prompt,
        negative_prompt,
        grow_size,
        edge_strength,
        pose_strength,
        depth_strength,
        inpaint_strength,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler
    )
    global AUTO_SAVE
    if AUTO_SAVE:
        auto_save_generated_image(final_pil_image)
    return final_pil_image

def auto_save_generated_image(pil_image):
    os.makedirs("output", exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join("output", f"magicquill_{timestamp}.png")
    pil_image.save(save_path)
    print(f"Image saved to: {save_path}")

css = '''
    .row {
        width: 90%;
        margin: auto;
    }
    footer {
        visibility: 
        hidden
    }
    '''

with gr.Blocks(css=css) as demo:
    with gr.Row(elem_classes="row"):
        with gr.Column():
            original_image = gr.Image(type="filepath", label="Original Image")
            text_prompt = gr.Textbox(label="Text Prompt for Mask Generation", value="shirt", placeholder="Enter text to describe the object you want to segment")
            positive_prompt = gr.Textbox(label="Positive Prompt", value="", placeholder="Enter positive prompt for image generation")
            mask_btn = gr.Button("Generate Mask", variant="secondary")
        with gr.Column():
            output_image = gr.Image(type="filepath", label="Generated Image")
    with gr.Row(elem_classes="row"):
        with gr.Column():
            btn = gr.Button("Run", variant="primary")
        with gr.Column():
            with gr.Accordion("parameters", open=False):
                ckpt_name = gr.Dropdown(
                    label="Base Model Name",
                    choices=folder_paths.get_filename_list("checkpoints"),
                    value=os.path.join('SD1.5', 'realisticVisionV60B1_v51VAE.safetensors'),
                    interactive=True
                )
                auto_save_checkbox = gr.Checkbox(
                    label="Auto Save",
                    value=False,
                    interactive=True
                )
                resolution_slider = gr.Slider(
                    label="Resolution (Please update this before you upload the image)",
                    minimum=256,
                    maximum=2048,
                    value=512,
                    step=64,
                    interactive=True
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="",
                    interactive=True
                )
                grow_size = gr.Slider(
                    label="Grow Size",
                    minimum=0,
                    maximum=100,
                    value=15,
                    step=1,
                    interactive=True
                )
                edge_strength = gr.Slider(
                    label="Edge Strength",
                    minimum=0.0,
                    maximum=5.0,
                    value=0.55,
                    step=0.01,
                    interactive=True
                )
                pose_strength = gr.Slider(
                    label="Pose Strength",
                    minimum=0.0,
                    maximum=5.0,
                    value=0.55,
                    step=0.01,
                    interactive=True
                )
                depth_strength = gr.Slider(
                    label="Depth Strength",
                    minimum=0.0,
                    maximum=5.0,
                    value=0.55,
                    step=0.01,
                    interactive=True
                )
                inpaint_strength = gr.Slider(
                    label="Inpaint Strength",
                    minimum=0.0,
                    maximum=5.0,
                    value=1.0,
                    step=0.01,
                    interactive=True
                )
                seed = gr.Number(
                    label="Seed",
                    value=-1,
                    precision=0,
                    interactive=True
                )
                steps = gr.Slider(
                    label="Steps",
                    minimum=1,
                    maximum=50,
                    value=20,
                    interactive=True
                )
                cfg = gr.Slider(
                    label="CFG",
                    minimum=0.0,
                    maximum=100.0,
                    value=5.0,
                    step=0.1,
                    interactive=True
                )
                sampler_name = gr.Dropdown(
                    label="Sampler Name",
                    choices=["euler", "euler_ancestral", "heun", "heunpp2","dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ddim", "uni_pc", "uni_pc_bh2"],
                    value='euler_ancestral',
                    interactive=True
                )
                scheduler = gr.Dropdown(
                    label="Scheduler",
                    choices=["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"],
                    value='karras',
                    interactive=True
                )

        def update_auto_save(value):
            global AUTO_SAVE
            AUTO_SAVE = value
            
        def update_resolution(value):
            global RES
            RES = value
            # Re-cache original image with new resolution if available
            global ORIGINAL_IMAGE_TENSOR, original_image
            if original_image.value is not None:
                ORIGINAL_IMAGE_TENSOR = load_and_resize_image(original_image.value, max_size=RES)

        auto_save_checkbox.change(fn=update_auto_save, inputs=[auto_save_checkbox])
        resolution_slider.change(fn=update_resolution, inputs=[resolution_slider])
        mask_btn.click(generate_mask_handler, inputs=[original_image, text_prompt], outputs=[original_image])
        btn.click(generate_image_handler, inputs=[original_image, text_prompt, positive_prompt, ckpt_name, negative_prompt, grow_size, edge_strength, pose_strength, depth_strength, inpaint_strength, seed, steps, cfg, sampler_name, scheduler], outputs=[output_image])
        
        # Make the output image the same size as the input image
        original_image.change(fn=lambda x: gr.update(height=512, width=512) if x is not None else None, inputs=[original_image], outputs=[output_image])
    
if __name__ == "__main__":
    demo.launch(share=True, server_port=7860)