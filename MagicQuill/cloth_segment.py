from transformers import pipeline
from PIL import Image
import numpy as np



class ClothSegment:
    def __init__(self):
        self.segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes")

    def generate_mask(self, img, clothes = ["Hat", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Scarf"]):
        # Segment image
        segments = self.segmenter(img)

        # Create list of masks
        mask_list = []
        for s in segments:
            if(s['label'] in clothes):
                mask_list.append(s['mask'])

        # Check if any masks were found
        if not mask_list:
            # Return empty mask if no clothes were detected
            width, height = img.size
            final_mask = Image.new('L', (width, height), 0)
            return final_mask, img

        # Paste all masks on top of eachother 
        final_mask = np.array(mask_list[0])
        for mask in mask_list[1:]:  # Start from index 1 to avoid reprocessing the first mask
            current_mask = np.array(mask)
            final_mask = np.logical_or(final_mask, current_mask).astype(np.uint8)  # Use logical OR for better mask combination
                
        # Convert final mask from np array to PIL image
        final_mask = Image.fromarray(final_mask * 255, mode='L')

        # Apply mask to original image
        img.putalpha(final_mask)

        return final_mask, img
