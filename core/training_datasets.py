from math import inf

import torch
import os
import glob
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
import torchvision.transforms.functional as TF 
import numpy as np
import random
from PIL import Image
from pathlib import Path
from .utils import frame_utils
import logging
import cv2
from .augmentor import StereoAugmentor
import scipy.signal 
class TrainingDataset(Dataset):
    def __init__(self, reader = None, augmentor = None, is_phase_2=False, sparse = False):
        # Function from frame_utils to read the disparity
        if reader:
            self.disparity_reader = reader
        else:
            self.disparity_reader = frame_utils.read_gen
        
        self.left_img_paths = []
        self.right_img_paths = []
        self.disp_paths = []
        
        self.max_disp = 192
        
        self.augmentor = augmentor
        
        self.is_phase_2 = is_phase_2
        
        self.sparse = sparse
    
    def __getitem__(self, index):
        print(self.left_img_paths[index], self.right_img_paths[index], self.disp_paths[index])
        
        img1 = frame_utils.read_gen(self.left_img_paths[index])
        img2 = frame_utils.read_gen(self.right_img_paths[index])
        disp = self.disparity_reader(self.disp_paths[index]) # np.float32
        
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        disp = np.array(disp).astype(np.float32)
        
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < self.max_disp
            
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]

        if len(img2.shape) == 2:
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img2 = img2[..., :3]
        
        if self.augmentor:
            clean_img1, clean_img2, aug_img1, aug_img2, disp, valid = self.augmentor(img1, img2, disp, valid, self.is_phase_2)
        else:
            clean_img1, clean_img2 = img1, img2
            aug_img1, aug_img2 = img1, img2
        
        # Convert to torch tensors
        clean_img1 = torch.from_numpy(np.ascontiguousarray(clean_img1)).permute(2, 0, 1).float()
        clean_img2 = torch.from_numpy(np.ascontiguousarray(clean_img2)).permute(2, 0, 1).float()
        
        aug_img1 = torch.from_numpy(np.ascontiguousarray(aug_img1)).permute(2, 0, 1).float()
        aug_img2 = torch.from_numpy(np.ascontiguousarray(aug_img2)).permute(2, 0, 1).float()
        
        disp = torch.from_numpy(disp).unsqueeze(0)
        valid = torch.from_numpy(valid).float()
        return clean_img1, clean_img2, aug_img1,aug_img2, disp, valid

    def __len__(self):
        return len(self.left_img_paths)

class SceneFlowDataset(TrainingDataset):
    def __init__(self, root_dir = './data/datasets/SceneFlow', augmentor = None, is_phase_2 = False, mode = 'TRAIN', subsets=['flyingthings', 'driving', 'monkaa']):
        super().__init__(augmentor=augmentor, is_phase_2 = is_phase_2)
        self.root_dir = root_dir
        self.mode = mode.upper()
        
        for subset in subsets:
            if subset.lower() == 'flyingthings':
                self._load_flying()
            elif subset.lower() == 'driving':
                self._load_driving()
            elif subset.lower() == 'monkaa':
                self._load_monkaa()
                
    def _load_flying(self):
        search_pattern = os.path.join(self.root_dir, 'FlyingThings3D', f'frames_cleanpass/{self.mode}/*/*/left/*.png')
        images = sorted(glob.glob(search_pattern))
        
        self.left_img_paths.extend(images)
        self.right_img_paths.extend([p.replace('left', 'right') for p in self.left_img_paths])
        self.disp_paths.extend([p.replace('frames_cleanpass', 'disparity').replace('.png', '.pfm') for p in self.left_img_paths])
        
    def _load_driving(self):
        if self.mode == 'TRAIN':
            search_pattern = os.path.join(self.root_dir, 'Driving', 'frames_cleanpass', '**/*.png')
            images = sorted(glob.glob(search_pattern, recursive=True))
            
            left_images = [p for p in images if '/left/' in p]
            self.left_img_paths.extend(left_images)
            self.right_img_paths.extend([p.replace('/left/', '/right/') for p in left_images])
            self.disp_paths.extend([p.replace('frames_cleanpass', 'disparity').replace('.png', '.pfm') for p in left_images])
    
    def _load_monkaa(self):
        if self.mode == 'TRAIN':
            search_pattern = os.path.join(self.root_dir, 'Monkaa', 'frames_cleanpass', '**/*.png')
            all_images = sorted(glob.glob(search_pattern, recursive=True))
            
            left_images = [p for p in all_images if '/left/' in p]
            
            self.left_img_paths.extend(left_images)
            self.right_img_paths.extend([p.replace('/left/', '/right/') for p in left_images])
            self.disp_paths.extend([p.replace('frames_cleanpass', 'disparity').replace('.png', '.pfm') for p in left_images])



class TartanAir(TrainingDataset):
    def __init__(self, root_dir = '', augmentor = None, is_phase_2 = False):
        super().__init__(reader=frame_utils.readDispTartanAir, augmentor = augmentor, is_phase_2 = is_phase_2)
        
        search_pattern = os.path.join(root_dir, 'image_left/*.png')
        self.left_paths = sorted(glob.glob(search_pattern))
        self.right_paths = [p.replace('image_left', 'image_right') for p in self.left_paths]
        self.disp_paths = [p.replace('image_left', 'depth_left').replace('.png', '_depth.npy') for p in self.left_paths]


class Middlebury(TrainingDataset):
    def __init__(self, root='./data/datasets/Middlebury', split='2014', resolution='F', augmentor=None, is_phase_2=False):
        super(Middlebury, self).__init__(reader=frame_utils.readDispMiddlebury, augmentor=augmentor, is_phase_2=is_phase_2)
        assert os.path.exists(root)
        assert split in ["2005", "2006", "2014", "2021", "MiddEval3"]
        if split == "2005":
            scenes = list((Path(root) / "2005").glob("*"))
            for scene in scenes:
                path = os.path.join(scene, "view1.png")
                self.left_img_paths.extend([path])
                self.right_img_paths.extend([os.path.join(scene, "view5.png")])
                self.disp_paths.extend([os.path.join(scene, "disp1.png")])
                
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:
                        self.left_img_paths.extend([os.path.join(scene, f"Illum{illum}/Exp{exp}/view1.png")])
                        self.right_img_paths.extend([os.path.join(scene, f"Illum{illum}/Exp{exp}/view5.png")])
                        self.disp_paths.extend([os.path.join(scene, f"disp1.png")])
        
        elif split == "2006":
            scenes = list((Path(root) / "2006").glob("*"))
            for scene in scenes:
                path = os.path.join(scene, "view1.png")
                self.left_img_paths.extend([path])
                self.right_img_paths.extend([os.path.join(scene, "view1.png")])
                self.disp_paths.extend([os.path.join(scene, "disp1.png")])
                
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:
                        self.left_img_paths.extend([os.path.join(scene, f"Illum{illum}/Exp{exp}/view1.png")])
                        self.right_img_paths.extend([os.path.join(scene, f"Illum{illum}/Exp{exp}/view5.png")])
                        self.disp_paths.extend([os.path.join(scene, f"/disp1.png")])
        elif split == "2014":
            scenes = list((Path(root) / "2014").glob("*"))
            for scene in scenes:
                for s in ["E", "L", ""]:
                    self.left_img_paths.extend([os.path.join(scene, "im0.png")])
                    self.right_img_paths.extend([os.path.join(scene, f"im1{s}.png")])
                    self.disp_paths.extend([os.path.join(scene, f"disp0.pfm")])
        
        elif split == "2021":
            scenes = list((Path(root) / "2021/data").glob("*"))
            for scene in scenes:
                path = os.path.join(scene, "im0.png")
                self.left_img_paths.extend([path])
                self.right_img_paths.extend([os.path.join(scene, "im1.png")])
                self.disp_paths.extend([os.path.join(scene, "disp0.pfm")])
                
                for s in ["0", "1", "2", "3"]:
                    if os.path.exists(str(scene / f"ambient/L0/im0e{s}.png")):
                        path = str(scene / f"ambient/L0/im0e{s}.png")
                        self.left_img_paths.extend([path])
                        self.right_img_paths.extend([path.replace(f"im0e{s}", f"im1e{s}")])
                        self.disp_paths.extend([path.replace(f"im0e{s}.png", "disp0.pfm")])
        else:
            search_pattern = os.path.join(root, "MiddEval3", f'training{resolution}', '*/im0.png')
            self.left_img_paths = sorted(glob.glob(search_pattern))
            self.right_img_paths = [p.replace('im0', 'im1') for p in self.left_img_paths]
            self.disp_paths = [p.replace('im0.png', 'disp0GT.pfm') for p in self.left_img_paths]
            assert len(self.left_img_paths) == len(self.right_img_paths) == len(self.disp_paths) > 0, [self.left_img_paths, split]

class ETH3D(TrainingDataset):
    def __init__(self, root_dir = './data/datasets/ETH3D', augmentor = None, split = 'training'):
        super().__init__(reader='', augmentor=augmentor)
        
        search_pattern = os.path.join(root_dir, f'two_view_{split}', '*/im0.png')
        self.left_img_paths = sorted(glob.glob(search_pattern))
        self.right_img_paths = [p.replace('im0', 'im1') for p in self.left_img_paths]
        
        disp_pattern = os.path.join(root_dir, f'two_view_{split}_gt', '*/disp0GT.pfm')
        self.disp_paths = sorted(glob.glob(disp_pattern))
        
        self.occ_mask = [p.replace('disp0GT.pfm', 'mask0nocc.png') for p in self.disp_paths]
        
    def __getitem__(self, index):
        clean_img1, clean_img2, aug_img1,aug_img2, disp, valid = super().__getitem__(index)

        occ_file = self.occ_mask[index]
        return clean_img1, clean_img2, aug_img1,aug_img2, disp, valid, occ_file

def fetch_training_dataloader(is_phase_2):
    
    stereo_augmentor = StereoAugmentor(crop_size=(256, 512), apply_clr_jitter=True)
    
    scene_flow = SceneFlowDataset(augmentor=stereo_augmentor, is_phase_2=is_phase_2, mode="TRAIN")
    datasets = []
    datasets.append(scene_flow)
    
    # Add Middlebury datasets for training
    # for split in ['2005', '2006', '2021']:
    #     datasets.append(Middlebury(
    #         augmentor=stereo_augmentor, 
    #         is_phase_2=is_phase_2, 
    #         split=split
    #     ))
    
    training_dataset = ConcatDataset(datasets)

    
    logging.info(f"Training with {len(training_dataset)} image pairs")
    print(f"Training with {len(training_dataset)} image pairs")
    
    train_loader = DataLoader(
        training_dataset,
        batch_size = 8,
        shuffle = True,
        num_workers = 8,
        pin_memory = True,
        drop_last = True
    )
    return train_loader

def fetch_testing_dataloader(dataset = 'sceneflow'):
    
    if dataset == 'sceneflow':
        val_dataset = SceneFlowDataset(
            augmentor=None,
            is_phase_2=False,
            mode='TEST',
            subsets=['flyingthings']
        )
    # elif dataset == 'middlebury':
    #     val_dataset = Middlebury(
    #         augmentor=None,
    #         is_phase_2=False,
    #         split='MiddEval3',
    #         resolution='F'
    #     )
    # elif dataset == 'eth3d':
    #     val_dataset = ETH3D(augmentor=None)
        
    val_loader = DataLoader(
        val_dataset,
        batch_size = 4,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    return val_loader

def fetch_hard_testing_samples():
    dataset = SceneFlowDataset(augmentor=None, is_phase_2=False, mode='TEST', subsets=['flyingthings'])
    # for idx in range(len(dataset)):
    idx = 3237
    img1, img2, _, _, _, _ = dataset[idx]

    # Convert to numpy
    img1_np = img1.permute(1, 2, 0).contiguous().cpu().numpy()
    img2_np = img2.permute(1, 2, 0).contiguous().cpu().numpy()
    
    # Change to gray scale
    img1_gray = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)
    
    height, width = img1_gray.shape
    patch_size = 32
    row_start = [i for i in range(0, height, patch_size)]
    col_start = [i for i in range(0, width, patch_size)]
    
    
    indices = [ (i, j) for i in row_start for j in col_start if i + patch_size <= height and j + patch_size <= width]
    patch_score = []
    for row, col in indices:
        # Only interested in horizontal pixel correlations
        patch1 = img1_gray[row:row+patch_size, col:col+patch_size]
        patch1_1d = np.mean(patch1, axis=0)
        
        # Subtract mean to center the data
        patch1_1d = patch1_1d - np.mean(patch1_1d)  
        corr = np.correlate(patch1_1d, patch1_1d, mode='same')
        corr = corr / np.max(corr)  # Normalize the correlation
        corr[len(corr)//2-4:len(corr)//2+4] = -np.inf  # Zero out the central peak to avoid trivial correlation
        # corr = np.concat([corr[:patch_size//2], corr[patch_size//2+1:]]) 
        peaks = scipy.signal.find_peaks(corr, height=(0.0, 1))[0]
        peak_values = corr[peaks]
        score = np.max(peak_values) if peak_values.size > 0 else -np.inf
        
        patch_score.append((score, row, col))
        
    s = [s for s,r,c in patch_score if r==96 and c==896]
    print(s)
    patch_score.sort(key=lambda x: x[0])
    min_score = patch_score[0]
    mid_score = patch_score[len(patch_score)//2]
    max_score = patch_score[-1]
    # print(patch_score)
    p_min = img1_np[min_score[1]:min_score[1]+patch_size, min_score[2]:min_score[2]+patch_size]
    p_mid = img1_np[mid_score[1]:mid_score[1]+patch_size, mid_score[2]:mid_score[2]+patch_size]
    p_max = img1_np[max_score[1]:max_score[1]+patch_size, max_score[2]:max_score[2]+patch_size]
    
    img = cv2.rectangle(img1_np.copy().astype(np.uint8), (min_score[2], min_score[1]), (min_score[2]+patch_size, min_score[1]+patch_size), (255, 0, 0), 2)
    img = cv2.rectangle(img.copy(), (mid_score[2], mid_score[1]), (mid_score[2]+patch_size, mid_score[1]+patch_size), (0, 255, 0), 2)
    img = cv2.rectangle(img.copy(), (max_score[2], max_score[1]), (max_score[2]+patch_size, max_score[1]+patch_size), (0, 0, 255), 2)
    img = cv2.rectangle(img.copy(), (896, 96), (896+patch_size, 96+patch_size), (255, 255, 0), 2)
    cv2.imwrite(f'./hard_testing_samples/img_with_patches_{idx}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    img_min = Image.fromarray(p_min.astype(np.uint8)).save(f'./hard_testing_samples/img_min_{idx}.png')
    img_mid = Image.fromarray(p_mid.astype(np.uint8)).save(f'./hard_testing_samples/img_mid_{idx}.png')
    img_max = Image.fromarray(p_max.astype(np.uint8)).save(f'./hard_testing_samples/img_max_{idx}.png')
        # break
    
def test_middlebury():
    dataset = Middlebury(split='2005', augmentor=None, is_phase_2=False)
    img1, img2, _, _, _, _ = dataset[1]
    # cv2.imwrite('img1.png', img1.permute(1, 2, 0).contiguous().cpu().numpy())
    
if __name__ == '__main__':
    test_middlebury()