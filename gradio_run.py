import os
import gradio as gr
import random
import torch
import numpy as np
from PIL import Image, ImageOps
import base64
import io
from MagicQuill import folder_paths
from MagicQuill.scribble_color_edit import ScribbleColorEditModel
from MagicQuill.grounded_segment_anything import GroundedSegmentAnything
import time
import io

AUTO_SAVE = False
RES = 512

scribbleColorEditModel = ScribbleColorEditModel()
groundedSegmentAnything = GroundedSegmentAnything()

def tensor_to_base64(tensor):
    tensor = tensor.squeeze(0) * 255.
    pil_image = Image.fromarray(tensor.cpu().byte().numpy())
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_str

def read_base64_image(base64_image):
    if base64_image.startswith("data:image/png;base64,"):
        base64_image = base64_image.split(",")[1]
    elif base64_image.startswith("data:image/jpeg;base64,"):
        base64_image = base64_image.split(",")[1]
    elif base64_image.startswith("data:image/webp;base64,"):
        base64_image = base64_image.split(",")[1]
    else:
        raise ValueError("Unsupported image format.")
    image_data = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_data))
    image = ImageOps.exif_transpose(image)
    return image

def create_alpha_mask(base64_image):
    """Create an alpha mask from the alpha channel of an image."""
    image = read_base64_image(base64_image)
    mask = torch.zeros((1, image.height, image.width), dtype=torch.float32, device="cpu")
    if 'A' in image.getbands():
        alpha_channel = np.array(image.getchannel('A')).astype(np.float32) / 255.0
        mask[0] = 1.0 - torch.from_numpy(alpha_channel)
    return mask

def load_and_preprocess_image(base64_image, convert_to='RGB'):
    """Load and preprocess a base64 image."""
    image = read_base64_image(base64_image)
    image = image.convert(convert_to)
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array)[None,]
    return image_tensor

def load_and_resize_image(base64_image, convert_to='RGB', max_size=512):
    """Load and preprocess a base64 image, resize if necessary."""
    image = read_base64_image(base64_image)
    image = image.convert(convert_to)
    width, height = image.size
    scaling_factor = max_size / min(width, height)
    new_size = (int(width * scaling_factor), int(height * scaling_factor))
    image = image.resize(new_size, Image.LANCZOS)
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array)[None,]
    return image_tensor

def prepare_images_and_masks(original_image, text_prompt):
    original_image_tensor = load_and_preprocess_image(original_image)
    # Generate mask using Grounded Segment Anything
    image_pil = read_base64_image(original_image)
    mask_pil, annotated_pil = groundedSegmentAnything.generate_mask(image_pil, text_prompt)
    # Convert mask to tensor
    mask_array = np.array(mask_pil).astype(np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_array)[None,]
    # Convert annotated image to base64
    buffered = io.BytesIO()
    annotated_pil.save(buffered, format="PNG")
    annotated_img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    annotated_img_base64 = "data:image/png;base64," + annotated_img_str
    return original_image_tensor, mask_tensor, annotated_img_base64


def generate(ckpt_name, original_image, text_prompt, positive_prompt, negative_prompt, grow_size, fine_edge, edge_strength, pose_strength, depth_strength, inpaint_strength, seed, steps, cfg, sampler_name, scheduler):
    original_image, mask, annotated_image = prepare_images_and_masks(original_image, text_prompt)
    progress = None
    if fine_edge == 'enable':
        edge_strength *= 2.0

    latent_samples, final_image, lineart_output, pose_output, depth_output = scribbleColorEditModel.process(
        ckpt_name,
        original_image, 
        positive_prompt, 
        negative_prompt, 
        mask, 
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
        progress
    )

    final_image_base64 = tensor_to_base64(final_image)
    return final_image_base64, annotated_image

def generate_image_handler(original_image, text_prompt, ckpt_name, negative_prompt, fine_edge, grow_size, edge_strength, pose_strength, depth_strength, inpaint_strength, seed, steps, cfg, sampler_name, scheduler):
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    res, annotated_image = generate(
        ckpt_name,
        original_image,
        text_prompt,
        positive_prompt,  # This should be defined somewhere or passed as a parameter
        negative_prompt,
        grow_size,
        fine_edge,
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
        auto_save_generated_image(res)
    return res, annotated_image

def auto_save_generated_image(res):
    img_str = res
    if img_str.startswith("data:image/png;base64,"):
        img_str = img_str.split(",")[1]
    img_data = base64.b64decode(img_str)
    img = Image.open(io.BytesIO(img_data))
    
    os.makedirs("output", exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join("output", f"magicquill_{timestamp}.png")
    img.save(save_path)
    print(f"Image saved to: {save_path}")

positive_prompt = ""  # Define positive_prompt as a global variable

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
            original_image = gr.Image(type="base64", label="Original Image")
            text_prompt = gr.Textbox(label="Text Prompt for Mask Generation", value="shirt", placeholder="Enter text to describe the object you want to segment")
        with gr.Column():
            output_image = gr.Image(type="base64", label="Generated Image")
            annotated_image = gr.Image(type="base64", label="Annotated Image")
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
                    label="Resolution (Please update this before you upload the image ;).)",
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
                fine_edge = gr.Radio(
                    label="Fine Edge",
                    choices=['enable', 'disable'],
                    value='disable',
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

        auto_save_checkbox.change(fn=update_auto_save, inputs=[auto_save_checkbox])
        resolution_slider.change(fn=update_resolution, inputs=[resolution_slider])
        btn.click(generate_image_handler, inputs=[original_image, text_prompt, ckpt_name, negative_prompt, fine_edge, grow_size, edge_strength, pose_strength, depth_strength, inpaint_strength, seed, steps, cfg, sampler_name, scheduler], outputs=[output_image, annotated_image])
    
if __name__ == "__main__":
    demo.launch(share=True, server_port=8080)