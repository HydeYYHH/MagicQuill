import numpy as np
import torch
import torchvision
import cv2
from PIL import Image
import supervision as sv
import folder_paths

import sys
sys.path.append('./Grounded-Segment-Anything')
sys.path.append('./Grounded-Segment-Anything/GroundingDINO')
sys.path.append('./Grounded-Segment-Anything/segment_anything')

from groundingdino.util.inference import Model as GroundingDINOModel
from segment_anything import sam_model_registry, SamPredictor


class GroundedSegmentAnything:
    def __init__(self):
        grounding_dino_config_path = './Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        grounding_dino_path = folder_paths.get_full_path("grounded_segment_anything", "groundingdino_swint_ogc.pth")
        sam_checkpoint_path = folder_paths.get_full_path("grounded_segment_anything", "sam_vit_l_0b3195.pth")
        sam_model_type = 'vit_l'
        self.grounding_dino_model = GroundingDINOModel(
            model_config_path=grounding_dino_config_path,
            model_checkpoint_path=grounding_dino_path,
            device="cuda"
        )
        self.sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
        self.sam_model.to(device="cuda")
        self.sam_predictor = SamPredictor(self.sam_model)

    def generate_mask(self, image, text_prompt, box_threshold: float = 0.3, text_threshold: float = 0.3, nms_threshold: float = 0.8):
        if isinstance(image, Image.Image):
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            image_cv = image

        # --- GroundingDINO Detection ---
        detections = self.grounding_dino_model.predict_with_classes(
            image=image_cv,
            classes=[text_prompt],
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        # --- Non-Maximum Suppression (NMS) and Filtering ---
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            nms_threshold
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx] if detections.class_id is not None else None

        # If multiple detections remain, select the one with the highest confidence
        if len(detections.xyxy) > 1:
            best_idx = np.argmax(detections.confidence)
            detections.xyxy = detections.xyxy[best_idx:best_idx+1]
            detections.confidence = detections.confidence[best_idx:best_idx+1]
            detections.class_id = detections.class_id[best_idx:best_idx+1] if detections.class_id is not None else None

        # --- SAM Segmentation ---
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_rgb)

        # Combine all detected target masks
        combined_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        result_masks = []

        for box in detections.xyxy:
            masks, scores, _ = self.sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            # Select the best mask based on the score
            best_mask = masks[np.argmax(scores)]
            combined_mask = np.logical_or(combined_mask, best_mask).astype(np.uint8)
            result_masks.append(best_mask)

        # Convert combined mask to a PIL image
        mask_pil = Image.fromarray(combined_mask * 255, mode='L')

        # --- Annotation and Visualization ---
        detections.mask = np.array(result_masks)
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{text_prompt} {confidence:0.2f}"
            for confidence in detections.confidence
        ]
        annotated_image = mask_annotator.annotate(scene=image_cv.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # Convert annotated image back to PIL for display
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

        return mask_pil, annotated_pil