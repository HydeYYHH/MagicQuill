import os
import folder_paths
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.controlnet
import comfy.model_management
import torch
import torch.nn as nn
import numpy as np
import latent_preview
from PIL import Image
from einops import rearrange
import scipy.ndimage
import cv2
from magic_utils import HWC3, common_input_validate, resize_image_with_pad
from pidi import pidinet


supported_pt_extensions = set(['.ckpt', '.pt', '.bin', '.pth', '.safetensors', '.pkl'])
folder_names_and_paths = {}

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_path, "models")

folder_names_and_paths["checkpoints"] = ([os.path.join(models_dir, "checkpoints")], supported_pt_extensions)
folder_names_and_paths["configs"] = ([os.path.join(models_dir, "configs")], [".yaml"])

folder_names_and_paths["loras"] = ([os.path.join(models_dir, "loras")], supported_pt_extensions)
folder_names_and_paths["vae"] = ([os.path.join(models_dir, "vae")], supported_pt_extensions)
folder_names_and_paths["clip"] = ([os.path.join(models_dir, "clip")], supported_pt_extensions)
folder_names_and_paths["unet"] = ([os.path.join(models_dir, "unet")], supported_pt_extensions)
folder_names_and_paths["clip_vision"] = ([os.path.join(models_dir, "clip_vision")], supported_pt_extensions)
folder_names_and_paths["style_models"] = ([os.path.join(models_dir, "style_models")], supported_pt_extensions)
folder_names_and_paths["embeddings"] = ([os.path.join(models_dir, "embeddings")], supported_pt_extensions)
folder_names_and_paths["diffusers"] = ([os.path.join(models_dir, "diffusers")], ["folder"])
folder_names_and_paths["vae_approx"] = ([os.path.join(models_dir, "vae_approx")], supported_pt_extensions)

folder_names_and_paths["controlnet"] = ([os.path.join(models_dir, "controlnet"), os.path.join(models_dir, "t2i_adapter")], supported_pt_extensions)
folder_names_and_paths["gligen"] = ([os.path.join(models_dir, "gligen")], supported_pt_extensions)
folder_names_and_paths["upscale_models"] = ([os.path.join(models_dir, "upscale_models")], supported_pt_extensions)
folder_names_and_paths["hypernetworks"] = ([os.path.join(models_dir, "hypernetworks")], supported_pt_extensions)
folder_names_and_paths["photomaker"] = ([os.path.join(models_dir, "photomaker")], supported_pt_extensions)
folder_names_and_paths["classifiers"] = ([os.path.join(models_dir, "classifiers")], {""})

def common_annotator_call(model, tensor_image, input_batch=False, show_pbar=True, **kwargs):
    if "detect_resolution" in kwargs:
        del kwargs["detect_resolution"] #Prevent weird case?

    if "resolution" in kwargs:
        detect_resolution = kwargs["resolution"] if type(kwargs["resolution"]) == int and kwargs["resolution"] >= 64 else 512
        del kwargs["resolution"]
    else:
        detect_resolution = 512

    if input_batch:
        np_images = np.asarray(tensor_image * 255., dtype=np.uint8)
        np_results = model(np_images, output_type="np", detect_resolution=detect_resolution, **kwargs)
        return torch.from_numpy(np_results.astype(np.float32) / 255.0)

    batch_size = tensor_image.shape[0]
    if show_pbar:
        pbar = comfy.utils.ProgressBar(batch_size)
    out_tensor = None
    for i, image in enumerate(tensor_image):
        np_image = np.asarray(image.cpu() * 255., dtype=np.uint8)
        np_result = model(np_image, output_type="np", detect_resolution=detect_resolution, **kwargs)
        out = torch.from_numpy(np_result.astype(np.float32) / 255.0)
        if out_tensor is None:
            out_tensor = torch.zeros(batch_size, *out.shape, dtype=torch.float32)
        out_tensor[i] = out
        if show_pbar:
            pbar.update(1)
    return out_tensor

class CheckpointLoaderSimple:
    def load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        print("Loading checkpoint from:", ckpt_path)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out[:3]

class ControlNetLoader:
    def load_controlnet(self, control_net_name):
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = comfy.controlnet.load_controlnet(controlnet_path)
        return (controlnet, )

class ControlNetApplyAdvanced:
    def apply_controlnet(self, positive, negative, control_net, image, strength, start_percent, end_percent):
        if strength == 0:
            return (positive, negative)

        control_hint = image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])
    
class CLIPTextEncode:
    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], )

class KSampler:
    def common_ksampler(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
        latent_image = latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                    denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                    force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
        out = latent.copy()
        out["samples"] = samples
        return (out, )

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        return self.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)

class VAEDecode:
    def decode(self, vae, samples):
        return (vae.decode(samples["samples"]), )

norm_layer = nn.InstanceNorm2d
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features)
                        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        # Initial convolution block
        model0 = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    norm_layer(64),
                    nn.ReLU(inplace=True) ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model1 += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features//2
        for _ in range(2):
            model3 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [  nn.ReflectionPad2d(3),
                        nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out

def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z

class PidiNetDetector:
    def __init__(self, netNetwork):
        self.netNetwork = netNetwork
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, filename="table5_pidinet.pth"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, f"../models/preprocessor/{filename}")

        netNetwork = pidinet()
        netNetwork.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path)['state_dict'].items()})
        netNetwork.eval()

        return cls(netNetwork)

    def to(self, device):
        self.netNetwork.to(device)
        self.device = device
        return self
    
    def __call__(self, input_image, detect_resolution=512, safe=False, output_type="pil", scribble=False, apply_filter=True, upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        detected_map, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        
        detected_map = detected_map[:, :, ::-1].copy()
        with torch.no_grad():
            image_pidi = torch.from_numpy(detected_map).float().to(self.device)
            image_pidi = image_pidi / 255.0
            image_pidi = rearrange(image_pidi, 'h w c -> 1 c h w')
            edge = self.netNetwork(image_pidi)[-1]
            edge = edge.cpu().numpy()
            if apply_filter:
                edge = edge > 0.5 
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = edge[0, 0]

        if scribble:
            detected_map = nms(detected_map, 127, 3.0)
            detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
            detected_map[detected_map > 4] = 255
            detected_map[detected_map < 255] = 0

        detected_map = HWC3(remove_pad(detected_map))

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map

class GrowMask:
    def expand_mask(self, mask, expand, tapered_corners):
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c],
                           [1, 1, 1],
                           [c, 1, c]])
        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        out = []
        for m in mask:
            output = m.numpy()
            for _ in range(abs(expand)):
                if expand < 0:
                    output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                else:
                    output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            output = torch.from_numpy(output)
            out.append(output)
        return (torch.stack(out, dim=0),)
    
class PIDINET_Preprocessor:
    def execute(self, image, resolution=512, **kwargs):
        model = PidiNetDetector.from_pretrained().to(comfy.model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution, safe=True)
        del model
        return (out, )


import math
from collections import OrderedDict

def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_'+layer_name, nn.ReLU(inplace=True)))
    return nn.Sequential(OrderedDict(layers))

class bodypose_model(nn.Module):
    def __init__(self):
        super(bodypose_model, self).__init__()
        no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1', 'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2', 'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1', 'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L1']
        blocks = {}
        block0 = OrderedDict([
            ('conv1_1', [3, 64, 3, 1, 1]), ('conv1_2', [64, 64, 3, 1, 1]), ('pool1_stage1', [2, 2, 0]),
            ('conv2_1', [64, 128, 3, 1, 1]), ('conv2_2', [128, 128, 3, 1, 1]), ('pool2_stage1', [2, 2, 0]),
            ('conv3_1', [128, 256, 3, 1, 1]), ('conv3_2', [256, 256, 3, 1, 1]), ('conv3_3', [256, 256, 3, 1, 1]), ('conv3_4', [256, 256, 3, 1, 1]), ('pool3_stage1', [2, 2, 0]),
            ('conv4_1', [256, 512, 3, 1, 1]), ('conv4_2', [512, 512, 3, 1, 1]), ('conv4_3_CPM', [512, 256, 3, 1, 1]), ('conv4_4_CPM', [256, 128, 3, 1, 1])
        ])
        block1_1 = OrderedDict([
            ('conv5_1_CPM_L1', [128, 128, 3, 1, 1]), ('conv5_2_CPM_L1', [128, 128, 3, 1, 1]), ('conv5_3_CPM_L1', [128, 128, 3, 1, 1]),
            ('conv5_4_CPM_L1', [128, 512, 1, 1, 0]), ('conv5_5_CPM_L1', [512, 38, 1, 1, 0])
        ])
        block1_2 = OrderedDict([
            ('conv5_1_CPM_L2', [128, 128, 3, 1, 1]), ('conv5_2_CPM_L2', [128, 128, 3, 1, 1]), ('conv5_3_CPM_L2', [128, 128, 3, 1, 1]),
            ('conv5_4_CPM_L2', [128, 512, 1, 1, 0]), ('conv5_5_CPM_L2', [512, 19, 1, 1, 0])
        ])
        blocks['block1_1'] = block1_1
        blocks['block1_2'] = block1_2
        self.model0 = make_layers(block0, no_relu_layers)
        for i in range(2, 7):
            blocks[f'block{i}_1'] = OrderedDict([
                (f'Mconv1_stage{i}_L1', [185, 128, 7, 1, 3]), (f'Mconv2_stage{i}_L1', [128, 128, 7, 1, 3]), (f'Mconv3_stage{i}_L1', [128, 128, 7, 1, 3]),
                (f'Mconv4_stage{i}_L1', [128, 128, 7, 1, 3]), (f'Mconv5_stage{i}_L1', [128, 128, 7, 1, 3]), (f'Mconv6_stage{i}_L1', [128, 128, 1, 1, 0]), (f'Mconv7_stage{i}_L1', [128, 38, 1, 1, 0])
            ])
            blocks[f'block{i}_2'] = OrderedDict([
                (f'Mconv1_stage{i}_L2', [185, 128, 7, 1, 3]), (f'Mconv2_stage{i}_L2', [128, 128, 7, 1, 3]), (f'Mconv3_stage{i}_L2', [128, 128, 7, 1, 3]),
                (f'Mconv4_stage{i}_L2', [128, 128, 7, 1, 3]), (f'Mconv5_stage{i}_L2', [128, 128, 7, 1, 3]), (f'Mconv6_stage{i}_L2', [128, 128, 1, 1, 0]), (f'Mconv7_stage{i}_L2', [128, 19, 1, 1, 0])
            ])
        for k in blocks:
            blocks[k] = make_layers(blocks[k], no_relu_layers)
        
        self.model1_1, self.model2_1, self.model3_1, self.model4_1, self.model5_1, self.model6_1 = \
            blocks['block1_1'], blocks['block2_1'], blocks['block3_1'], blocks['block4_1'], blocks['block5_1'], blocks['block6_1']
        self.model1_2, self.model2_2, self.model3_2, self.model4_2, self.model5_2, self.model6_2 = \
            blocks['block1_2'], blocks['block2_2'], blocks['block3_2'], blocks['block4_2'], blocks['block5_2'], blocks['block6_2']

    def forward(self, x):
        out1 = self.model0(x)
        out1_1, out1_2 = self.model1_1(out1), self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)
        out2_1, out2_2 = self.model2_1(out2), self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)
        out3_1, out3_2 = self.model3_1(out3), self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)
        out4_1, out4_2 = self.model4_1(out4), self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)
        out5_1, out5_2 = self.model5_1(out5), self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)
        out6_1, out6_2 = self.model6_1(out6), self.model6_2(out6)
        return out6_1, out6_2

def draw_bodypose(canvas, candidate, subset):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1: continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index: continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX, mY = np.mean(X), np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas

# --- Main Rewritten Classes ---

class OpenPoseDetector:
    def __init__(self, model):
        self.model = model
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, filename="body_pose_model.pth"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "..", "models", "preprocessor", filename)
        if not os.path.exists(model_path):
            normalized_path = os.path.normpath(model_path)
            raise FileNotFoundError(
                f"OpenPose model file not found. "
                f"The script expected it at: '{normalized_path}'. "
                f"Please ensure the file '{filename}' exists in that location."
            )
        model = bodypose_model()
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_dict)
        model.eval()
        return cls(model)

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def __call__(self, input_image, detect_resolution=512, output_type="pil", upscale_method="INTER_CUBIC", **kwargs):
        input_image, _ = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        
        # Core OpenPose processing logic
        oriImg = input_image
        scale_search = [0.5]
        boxsize = 368
        stride = 8
        padValue = 128
        thre1, thre2 = 0.1, 0.05
        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
        paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

        for m in range(len(multiplier)):
            scale = multiplier[m]
            imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = resize_image_with_pad(imageToTest, boxsize, "INTER_CUBIC", pad_to_make_divisible=stride)
            
            im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 255 - 0.5
            im = np.ascontiguousarray(im)
            image = torch.from_numpy(im).float().to(self.device)

            with torch.no_grad():
                paf, heatmap = self.model(image)

            heatmap = heatmap.cpu().numpy()
            paf = paf.cpu().numpy()
            
            heatmap = np.transpose(np.squeeze(heatmap), (1, 2, 0))
            heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            paf = np.transpose(np.squeeze(paf), (1, 2, 0))
            paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            heatmap_avg += heatmap / len(multiplier)
            paf_avg += paf / len(multiplier)

        all_peaks, peak_counter = [], 0
        for part in range(18):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = scipy.ndimage.gaussian_filter(map_ori, sigma=3)
            map_left = np.zeros(one_heatmap.shape)
            map_left[1:, :] = one_heatmap[:-1, :]
            map_right = np.zeros(one_heatmap.shape)
            map_right[:-1, :] = one_heatmap[1:, :]
            map_up = np.zeros(one_heatmap.shape)
            map_up[:, 1:] = one_heatmap[:, :-1]
            map_down = np.zeros(one_heatmap.shape)
            map_down[:, :-1] = one_heatmap[:, 1:]
            peaks_binary = np.logical_and.reduce((one_heatmap >= map_left, one_heatmap >= map_right, one_heatmap >= map_up, one_heatmap >= map_down, one_heatmap > thre1))
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            peak_id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peaks))]
            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        connection_all, special_k = [], []
        mid_num = 10
        mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38], [45, 46]]
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]]
        for k in range(len(mapIdx)):
            score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
            candA = all_peaks[limbSeq[k][0] - 1]
            candB = all_peaks[limbSeq[k][1] - 1]
            nA, nB = len(candA), len(candB)
            if nA != 0 and nB != 0:
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        if norm == 0: continue
                        vec = np.divide(vec, norm)
                        startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), np.linspace(candA[i][1], candB[j][1], num=mid_num)))
                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] for I in range(len(startend))])
                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * oriImg.shape[0] / norm - 1, 0)
                        criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])
                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if len(connection) >= min(nA, nB): break
                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])
        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k]) - 1
                for i in range(len(connection_all[k])):
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1
                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][indexB] != partBs[i]:
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])
        
        subset = subset[subset[:, -1] >= 2]
        if len(subset) == 0: # Handle case with no detected bodies
             canvas = np.zeros_like(oriImg)
        else:
             subset = subset[subset[:, -2] / subset[:, -1] >= 0.2]
             canvas = np.zeros_like(oriImg)
             canvas = draw_bodypose(canvas, candidate, subset)

        detected_map = remove_pad(canvas)
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map

class OpenPose_Preprocessor:
    def execute(self, image, resolution=512, **kwargs):
        model = OpenPoseDetector.from_pretrained(filename="openpose.pth").to(comfy.model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution)
        del model
        return (out, )



class MiDaSDetector:
    def __init__(self, model):
        self.model = model
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, model_type="MiDaS_small"):
        model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
        model.eval()
        return cls(model)

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def __call__(self, input_image, detect_resolution=512, output_type="pil", upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        detected_map, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)

        with torch.no_grad():
            image = torch.from_numpy(detected_map).float().to(self.device)
            image = image / 255.0
            image = rearrange(image, 'h w c -> 1 c h w')
            depth = self.model(image)
            depth = depth.cpu().numpy()
            depth_min = depth.min()
            depth_max = depth.max()
            depth = (depth - depth_min) / (depth_max - depth_min)
            depth = (depth * 255.0).clip(0, 255).astype(np.uint8)
            depth = 255 - depth

        detected_map = HWC3(remove_pad(depth))
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map

class MiDaS_Preprocessor:
    def execute(self, image, resolution=512, **kwargs):
        model = MiDaSDetector.from_pretrained().to(comfy.model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution)
        del model
        return (out, )