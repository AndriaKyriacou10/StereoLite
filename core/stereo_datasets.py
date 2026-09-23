from math import inf

import torch
import os
import glob
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split, WeightedRandomSampler
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
        # print(f"{self.disp_paths[index]}")
        img1 = frame_utils.read_gen(self.left_img_paths[index])
        img2 = frame_utils.read_gen(self.right_img_paths[index])
        disp = self.disparity_reader(self.disp_paths[index]) # np.float32
                
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < self.max_disp
        
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        disp = np.array(disp).astype(np.float32)
        valid = valid & np.isfinite(disp) & (disp >= 0) & (disp < self.max_disp)
        
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
        
        # print(f"Index {index} | Left: {aug_img1.shape}, Right: {aug_img2.shape}, Disp: {disp.shape}")
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

class Middlebury(TrainingDataset):
    def __init__(self, root='./data/datasets/Middlebury', split='2014', resolution='F', augmentor=None, is_phase_2=False):
        super(Middlebury, self).__init__(reader=frame_utils.readDispMiddlebury, augmentor=augmentor, is_phase_2=is_phase_2)
        assert os.path.exists(root)
        assert split in ["2005", "2006", "2014", "2021", "MiddEval3"]
        if split == "2005":
            scenes = list((Path(root) / "2005").glob("*"))
            for scene in scenes:
                disp_path = os.path.join(scene, "disp1.png")
                if not os.path.exists(disp_path):
                    continue
                
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
                self.right_img_paths.extend([os.path.join(scene, "view5.png")])
                self.disp_paths.extend([os.path.join(scene, "disp1.png")])
                
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:
                        self.left_img_paths.extend([os.path.join(scene, f"Illum{illum}/Exp{exp}/view1.png")])
                        self.right_img_paths.extend([os.path.join(scene, f"Illum{illum}/Exp{exp}/view5.png")])
                        self.disp_paths.extend([os.path.join(scene, f"disp1.png")])
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
                        self.disp_paths.extend([os.path.join(scene, "disp0.pfm")])

        else:
            search_pattern = os.path.join(root, "MiddEval3", f'training{resolution}', '*/im0.png')
            self.left_img_paths = sorted(glob.glob(search_pattern))
            self.right_img_paths = [p.replace('im0', 'im1') for p in self.left_img_paths]
            self.disp_paths = [p.replace('im0.png', 'disp0GT.pfm') for p in self.left_img_paths]
            assert len(self.left_img_paths) == len(self.right_img_paths) == len(self.disp_paths) > 0, [self.left_img_paths, split]

class ETH3D(TrainingDataset):   
    def __init__(self, root_dir='./data/datasets/ETH3D', augmentor=None, condition='train', train_frac=0.5, seed=42, is_phase_2=False, return_occ = False):
        super().__init__(reader='', augmentor=augmentor, is_phase_2=is_phase_2)

        split = 'training'  # always — only pool with real GT
        search_pattern = os.path.join(root_dir, f'two_view_{split}', '*/im0.png')
        all_left = sorted(glob.glob(search_pattern))

        scene_ids = [os.path.basename(os.path.dirname(p)) for p in all_left]

        unique_scenes = sorted(set(scene_ids))
        rng = random.Random(seed)
        rng.shuffle(unique_scenes)
        split_point = int(len(unique_scenes) * train_frac)
        keep_scenes = set(unique_scenes[:split_point]) if condition == 'train' else set(unique_scenes[split_point:])

        self.left_img_paths = [p for p, sid in zip(all_left, scene_ids) if sid in keep_scenes]
        self.right_img_paths = [p.replace('im0', 'im1') for p in self.left_img_paths]
        self.disp_paths = [
            p.replace(f'two_view_{split}', f'two_view_{split}_gt').replace('im0.png', 'disp0GT.pfm')
            for p in self.left_img_paths
        ]    
        
        self.occ_mask = [p.replace('disp0GT.pfm', 'mask0nocc.png') for p in self.disp_paths]
        self.condition = condition
        self.return_occ = return_occ
    
    def __getitem__(self, index):
        clean_img1, clean_img2, aug_img1,aug_img2, disp, valid = super().__getitem__(index)

        if not self.return_occ:
            return clean_img1, clean_img2, aug_img1,aug_img2, disp, valid
        
        occ_file = self.occ_mask[index]
        return clean_img1, clean_img2, aug_img1,aug_img2, disp, valid, occ_file

def fetch_training_dataloader(is_phase_2, datasets = ['sceneflow'], middlebury_splits = ['2005', '2006', '2021'], sf_fraction = None, num_samples = None):
    
    stereo_augmentor = StereoAugmentor(crop_size=(256, 512), apply_clr_jitter=True)
    datasets_training = []
    is_sf = []
    for dataset in datasets:
        if dataset == 'sceneflow':
            scene_flow = SceneFlowDataset(augmentor=stereo_augmentor, is_phase_2=is_phase_2, mode="TRAIN")
            datasets_training.append(scene_flow)
            is_sf.append(True)
        elif dataset == 'eth3d':
            eth3d = ETH3D(augmentor=stereo_augmentor, condition='train', train_frac=0.5, is_phase_2=is_phase_2)
            datasets_training.append(eth3d)
            is_sf.append(False)
            
        elif dataset == 'middlebury':
            for split in middlebury_splits:
                middlebury = Middlebury(augmentor=stereo_augmentor, is_phase_2=is_phase_2, split=split)
                datasets_training.append(middlebury)
                is_sf.append(False)

    assert len(is_sf) == len(datasets_training)

    training_dataset = ConcatDataset(datasets_training)

    n_sf = sum(len(d) for d, s in zip(datasets_training, is_sf) if s)
    n_real = len(training_dataset) - n_sf
    logging.info(f"Training with {len(training_dataset)} image pairs")
    logging.info(f"Synthetic: {n_sf} | Real: {n_real}")
    
    
    train_loader = DataLoader(
        training_dataset,
        batch_size = 8,
        shuffle = True,
        num_workers = 8,
        pin_memory = True,
        drop_last = True
    )
    common = dict(batch_size=8, num_workers=8, pin_memory=True, drop_last=True)
    if sf_fraction is None:
        # Default: no weighted sampling, just shuffle the whole dataset
        return train_loader
    if n_sf == 0 or n_real == 0:
        logging.warning("sf_fraction ignored: only one of real/synthetic is present")
        return DataLoader(training_dataset, shuffle=True, **common)
    
    weights = []
    for d, s in zip(datasets_training, is_sf):
        w = sf_fraction / n_sf if s else (1.0 - sf_fraction) / n_real
        weights.extend([w] * len(d))
        
    assert len(weights) == len(training_dataset), f"{len(weights)} weights vs {len(training_dataset)} samples"
    
    n_draw = num_samples if num_samples is not None else len(training_dataset)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=n_draw, replacement=True)

    logging.info(f"weighted sampling: {100*(1-sf_fraction):.0f}% real / "
                 f"{100*sf_fraction:.0f}% synthetic, {n_draw} draws")
    # note: sampler and shuffle are mutually exclusive -- the sampler IS the shuffling
    return DataLoader(training_dataset, sampler=sampler, **common)

def fetch_testing_dataloader(datasets = ['sceneflow'], return_occ = False):
    val_datasets = []
    loaders = {}
    for dataset in datasets:
        
        if dataset == 'sceneflow':
            val_dataset = SceneFlowDataset(
                augmentor=None,
                is_phase_2=False,
                mode='TEST',
                subsets=['flyingthings']
            )
            # val_datasets.append(val_dataset)
            batch_size = 4
        elif dataset == 'middlebury':
            val_dataset = Middlebury(
                augmentor=None,
                is_phase_2=False,
                split='MiddEval3',
                resolution='F'
            )
            # val_datasets.append(val_dataset)
            batch_size = 1
        elif dataset == 'eth3d':
            val_dataset = ETH3D(augmentor=None, condition='test', return_occ=return_occ, is_phase_2=False)
            val_datasets.append(val_dataset)
            batch_size = 1
        
        loaders[dataset] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
    return loaders

    