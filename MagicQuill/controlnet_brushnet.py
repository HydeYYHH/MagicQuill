import torch.nn.functional as F
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .brushnet_nodes import BrushNetLoader, BrushNet, BlendInpaint, get_files_with_extension
from .comfyui_utils import CheckpointLoaderSimple, ControlNetLoader, ControlNetApplyAdvanced, CLIPTextEncode, KSampler, MiDaS_Preprocessor, OpenPose_Preprocessor, VAEDecode, GrowMask, PIDINET_Preprocessor

class ControlNetBrushNetModel():
    def __init__(self):
        self.checkpoint_loader = CheckpointLoaderSimple()
        self.clip_text_encoder = CLIPTextEncode()
        self.mask_processor = GrowMask()
        self.controlnet_loader = ControlNetLoader()
        self.scribble_processor = PIDINET_Preprocessor()
        self.pose_processor = OpenPose_Preprocessor()
        self.depth_processor = MiDaS_Preprocessor()
        self.brushnet_loader = BrushNetLoader()
        self.brushnet_node = BrushNet()
        self.controlnet_apply = ControlNetApplyAdvanced()
        self.ksampler = KSampler()
        self.vae_decoder = VAEDecode()
        self.blender = BlendInpaint()
        self.ckpt_name = os.path.join("SD1.5", "realisticVisionV60B1_v51VAE.safetensors")
        with torch.no_grad():
            self.model, self.clip, self.vae = self.checkpoint_loader.load_checkpoint(self.ckpt_name)
        self.load_models('SD1.5', 'float16')

    def load_models(self, base_model_version="SD1.5", dtype='float16'):
        if base_model_version == "SD1.5":
            edge_controlnet_name = "control_v11p_sd15_scribble.safetensors"
            pose_controlnet_name = "control_v11p_sd15_openpose.safetensors"
            depth_controlnet_name = "control_v11f1p_sd15_depth.safetensors"
            brushnet_name = os.path.join("brushnet", "random_mask_brushnet_ckpt", "diffusion_pytorch_model.safetensors")
        else:
            raise ValueError("Invalid base_model_version, not supported yet!!!: {}".format(base_model_version))
        self.edge_controlnet = self.controlnet_loader.load_controlnet(edge_controlnet_name)[0]
        self.pose_controlnet = self.controlnet_loader.load_controlnet(pose_controlnet_name)[0]
        self.depth_controlnet = self.controlnet_loader.load_controlnet(depth_controlnet_name)[0]
        
        self.brushnet_loader.inpaint_files = get_files_with_extension('inpaint')
        print("self.brushnet_loader.inpaint_files: ", get_files_with_extension('inpaint'))
        self.brushnet = self.brushnet_loader.brushnet_loading(brushnet_name, dtype)[0]
    
    def process(self, ckpt_name, image, positive_prompt, negative_prompt, mask, grow_size, edge_strength, pose_strength, depth_strength, inpaint_strength, seed, steps, cfg, sampler_name, scheduler, base_model_version='SD1.5', dtype='float16'):
        with torch.no_grad():
            if ckpt_name != self.ckpt_name:
                self.ckpt_name = ckpt_name
                self.model, self.clip, self.vae = self.checkpoint_loader.load_checkpoint(ckpt_name)
            if not hasattr(self, 'edge_controlnet') or not hasattr(self, 'pose_controlnet') or not hasattr(self, 'depth_controlnet') or not hasattr(self, 'brushnet'):
                self.load_models(base_model_version, dtype)
            
        positive = self.clip_text_encoder.encode(self.clip, positive_prompt)[0]
        negative = self.clip_text_encoder.encode(self.clip, negative_prompt)[0]        

        # Initialize output variables with None
        lineart_output = None
        pose_output = None
        depth_output = None

        mask = self.mask_processor.expand_mask(mask, expand=grow_size, tapered_corners=True)[0]

        if edge_strength > 0.0:
            print("Apply edge controlnet")
            # Resize masks to match the dimensions of lineart_output
            lineart_output = self.scribble_processor.execute(image, resolution=512)[0]
            mask_resized = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(1, lineart_output.shape[1], lineart_output.shape[2]), mode='nearest').squeeze(0).squeeze(0)
            bool_mask_resized = (mask_resized > 0.5)
            lineart_output[bool_mask_resized] = 0.0
            positive, negative = self.controlnet_apply.apply_controlnet(positive, negative, self.edge_controlnet, lineart_output, edge_strength, 0.0, 1.0)

        if pose_strength > 0.0:
            print("Apply pose controlnet")
            pose_output = self.pose_processor.execute(image, resolution=512)[0]
            positive, negative = self.controlnet_apply.apply_controlnet(positive, negative, self.pose_controlnet, pose_output, pose_strength, 0.0, 1.0)

        if depth_strength > 0.0:
            print("Apply depth controlnet")
            depth_output = self.depth_processor.execute(image, resolution=512)[0]
            positive, negative = self.controlnet_apply.apply_controlnet(positive, negative, self.depth_controlnet, depth_output, depth_strength, 0.0, 1.0)

        # BrushNet
        model, positive, negative, latent = self.brushnet_node.model_update(
            model=self.model,
            vae=self.vae,
            image=image,
            mask=mask,
            brushnet=self.brushnet,
            positive=positive,
            negative=negative,
            scale=inpaint_strength,
            start_at=0,
            end_at=10000
        )

        # KSampler Node
        latent_samples = self.ksampler.sample(
            model=model, 
            seed=seed, 
            steps=steps, 
            cfg=cfg, 
            sampler_name=sampler_name, 
            scheduler=scheduler, 
            positive=positive, 
            negative=negative, 
            latent_image=latent,
        )[0]

        # Image Blending
        final_image = self.vae_decoder.decode(self.vae, latent_samples)[0]
        final_image = self.blender.blend_inpaint(final_image, image, mask, kernel=10, sigma=10.0)[0]
        
        # Free up memory by deleting temporary variables
        del lineart_output, pose_output, depth_output, positive, negative, mask, image, latent_samples, model
        
        # Force garbage collection and empty CUDA cache
        torch.cuda.empty_cache()
        
        return (None, final_image, None, None, None)
