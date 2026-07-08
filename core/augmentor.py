from typing import Any

import numpy as np
import random 
import cv2 

class StereoAugmentor:
    def __init__(self, crop_size=(320, 736), apply_clr_jitter=True):
        self.crop_size = crop_size
        self.apply_clr_jitter = apply_clr_jitter
    
    def _brightness(self, img, b):
        return img * b
    
    def _contrast(self, img, contrast):
        mean = np.mean(img, axis = (0,1), keepdims=True)
        img = (img - mean)*contrast + mean
        return img
    
    def symmetric_color_jitter(self, img1, img2):
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        
        # Apply Brightness
        img1 = self._brightness(img1, brightness)
        img2 = self._brightness(img2, brightness)
        
        
        # Contrast
        img1 = self._contrast(img1, contrast)
        img2 = self._contrast(img2, contrast)
        
        return np.clip(img1, 0, 255).astype(np.uint8), np.clip(img2, 0, 255).astype(np.uint8) # make sure values of imgs remain between 0 and 255
    
    def asymmetric_color_jitter(self, img1, img2):
        # Generate independent random values for BOTH images
        b1, c1 = random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)
        b2, c2 = random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)
        
        # Brightness
        img1 = self._brightness(img1, b1)
        img2 = self._brightness(img2, b2)
        
        
        # Contrast
        img1 = self._contrast(img1, c1)
        img2 = self._contrast(img2, c2)
        
        return np.clip(img1, 0, 255).astype(np.uint8), np.clip(img2, 0, 255).astype(np.uint8)
    
    def apply_eraser(self, img, num_boxes=1):
        # To simulate occlusion, randomly erase regions in the image
        img_erased = img.copy()
        mean_color = np.mean(img_erased, axis=(0,1), keepdims=True)
        
        for _ in range(num_boxes):
            h, w = img.shape[:2]
            box_w = random.randint(50, 100) 
            box_h = random.randint(50, 100)
            
            x1 = random.randint(0, w - box_w)
            y1 = random.randint(0, h - box_h)
            
            # DIRECT ASSIGNMENT: Destroys the texture completely
            img_erased[y1:y1+box_h, x1:x1+box_w] = mean_color
            
        return img_erased
     
    def __call__(self, img1, img2, disp, valid, is_phase_2 = False):
        
        # Apply Synchronized Random Cropping
        h, w = img1.shape[:2]
        th, tw = self.crop_size
        
        # If the image is smaller than the crop size, we will pad it with zeros
        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)
        
        if pad_h > 0 or pad_w > 0:
            img1 = np.pad(img1, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
            img2 = np.pad(img2, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
            disp = np.pad(disp, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            valid = np.pad(valid, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            
            h, w = img1.shape[:2]
        
        x1 = random.randint(0, max(0, w - tw))
        y1 = random.randint(0, max(0, h - th))
        
        img1 = img1[y1:y1+th, x1:x1+tw]
        img2 = img2[y1:y1+th, x1:x1+tw]
        disp = disp[y1:y1+th, x1:x1+tw]
        valid = valid[y1:y1+th, x1:x1+tw]
        
        clean_img1 = img1.copy()
        clean_img2 = img2.copy()
        
        if is_phase_2 and random.random() < 0.20:
            aug_img1, aug_img2 = self.asymmetric_color_jitter(img1, img2)
            aug_img2 = self.apply_eraser(aug_img2, num_boxes=random.randint(1, 2))
        else:
            if hasattr(self, 'apply_clr_jitter') and self.apply_clr_jitter:
                aug_img1, aug_img2 = self.symmetric_color_jitter(img1, img2)
            else:
                aug_img1, aug_img2 = img1.copy(), img2.copy()
        
        return clean_img1, clean_img2, aug_img1, aug_img2, disp, valid        
        
        
                