import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import os
import copy
import random
from pathlib import Path
from glob import glob
import os.path as osp
from .utils import frame_utils
from collections import defaultdict
import json

class SceneFlowVideo():
    def __init__(self, root_dir='./data/datasets/SceneFlow', mode='TRAIN', subsets=['flyingthings', 'driving', 'monkaa'], max_disp = None):
        self.left_img_paths = []
        self.right_img_paths = []
        self.disp_paths = []
        self.flow_paths = []
        self.max_disp = max_disp
        self.root_dir = root_dir
        self.mode = mode.upper()
        self.scenes = []
        
        if 'flyingthings' in subsets:
            self._load_flying()
        
    def _load_flying(self):
        search_pattern = os.path.join(self.root_dir, 'FlyingThings3D', f'frames_cleanpass/{self.mode}/*/*/left/*.png')
        left_imgs = sorted(glob(search_pattern))
        
        self.left_img_paths.extend(left_imgs)
        # self.right_img_paths.extend([p.replace('left', 'right') for p in self.left_img_paths])
        # self.disp_paths.extend([p.replace('frames_cleanpass', 'disparity').replace('.png', '.pfm') for p in self.left_img_paths])
        
        scenes_dict = defaultdict(list)
        for left_path in self.left_img_paths:
            scene_id = os.path.dirname(os.path.dirname(left_path)).split(f'{self.mode}/')[-1] #e.g A/0000
            frame_num = os.path.basename(left_path).split('.')[0] # e.g 0006
            
            
            right_path = left_path.replace('left', 'right')
            disp_path = left_path.replace('frames_cleanpass', 'disparity').replace('.png', '.pfm')
            flow_path = os.path.join(
                            self.root_dir, 'FlyingThings3D', 'optical_flow', self.mode, scene_id,
                            'into_future', 'left', f'OpticalFlowIntoFuture_{frame_num}_L.pfm'
                        )                               
            
            if not (os.path.exists(right_path) and os.path.exists(disp_path)):
                print(f"[WARN] Skipping incomplete frame: {left_path}")
                continue
            frame_num = int(frame_num)
            scenes_dict[f'{scene_id}'].append({
                'frame_num': frame_num, 
                'left': left_path, 
                'right': right_path, 
                'disp': disp_path, 
                'flow': flow_path
            })
            
        for scene_id, frames in scenes_dict.items():
            frames_sorted = sorted(frames, key=lambda f: f['frame_num'])
            self.scenes.append({
                'scene_id': scene_id,
                'left':  [f['left']  for f in frames_sorted],
                'right': [f['right'] for f in frames_sorted],
                'disp':  [f['disp']  for f in frames_sorted],
                'flow':  [f['flow']  for f in frames_sorted],
            })

        self.scenes.sort(key=lambda s: s['scene_id'])
    
    def _load_driving(self):
        if self.mode == 'TRAIN':
            search_pattern = os.path.join(self.root_dir, 'Driving', 'frames_cleanpass', '**/*.png')
            images = sorted(glob(search_pattern, recursive=True))
            
            left_images = [p for p in images if '/left/' in p]
            self.left_img_paths.extend(left_images)
            self.right_img_paths.extend([p.replace('/left/', '/right/') for p in left_images])
            self.disp_paths.extend([p.replace('frames_cleanpass', 'disparity').replace('.png', '.pfm') for p in left_images])
    
    def _fetch_images(self, images_path):
        imgs = []
        for img_path in images_path:
            img = frame_utils.read_gen(img_path)
            img = np.array(img).astype(np.uint8)
            
            if len(img.shape) == 2:
                img = np.tile(img[..., None], (1, 1, 3))
            else:
                img = img[..., :3]
            img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
            imgs.append(img)
        return imgs
    
    def _fetch_disparity(self, disp_paths):
        disp_scene , valid_scene = [], []
        for disp_path in disp_paths:
            disp = frame_utils.read_gen(disp_path)
            if isinstance(disp, tuple):
                disp, valid = disp
            else:
                if self.max_disp is not None:
                    valid = disp < self.max_disp
                else:
                    valid = disp > -1e6
                
            disp = np.array(disp).astype(np.float32)
            
            disp = torch.from_numpy(disp).unsqueeze(0)
            valid = torch.from_numpy(valid).float()
            
            disp_scene.append(disp)
            valid_scene.append(valid)
        return disp_scene, valid_scene
    
    def _fetch_flow(self, flow_paths):
        flow_list = []
        
        for i, flow_path in enumerate(flow_paths):
            if i == len(flow_paths) - 1: # need N-1 optical flows 
                break
            flow = frame_utils.readPFM(flow_path)
            flow =  flow[:, :, :-1] # only need first two channels: u->horizontal displacement, v->vertical displacement
            flow = np.array(flow).astype(np.float32)
            flow = torch.from_numpy(np.ascontiguousarray(flow)).permute(2, 0, 1)
            flow_list.append(flow)
            
        return flow_list            
    
    def __getitem__(self, index):
        """Return one full scene with all frames and corresponding disparity and flow maps"""
        scene = self.scenes[index]
        left_paths, right_paths, disp_paths, flow_paths = scene['left'], scene['right'], scene['disp'], scene['flow']
        
        left_imgs = self._fetch_images(left_paths)
        right_imgs = self._fetch_images(right_paths)
        disp_scene, valid_scene = self._fetch_disparity(disp_paths)
        flow_scene = self._fetch_flow(flow_paths)
        
        return {'scene_id': scene['scene_id'], 'left': left_imgs, 'right': right_imgs, 'disp': disp_scene, 'valid':valid_scene, 'flow': flow_scene}
        
    def __len__(self):
        return len(self.scenes)

class SintelStereoVideo():
    """
    Loads MPI-Sintel Stereo sequences for video evaluation.
    Mirrors SceneFlowVideo's __getitem__ output shape.

    Key differences from SceneFlowVideo to keep in mind while filling this in:
    - Sintel's GT-labeled split is called "training" (not "test") -- see
      the earlier discussion, "test" has no public ground truth.
    - Each folder under {dstype}_left/ is already one scene -- no need to
      parse/group scene_id from a flat glob like _load_flying does.
    - Sequence length varies per scene (not fixed at 10 like FlyingThings3D).
    """

    def __init__(self, root_dir='./data/datasets/Sintel', mode='training', dstype='clean', max_disp=None):
        self.root_dir = root_dir
        self.mode = mode
        self.dstype = dstype  # 'clean' or 'final'
        self.scenes = []
        self.max_disp = max_disp
        self._load_sintel()

    def _load_sintel(self):
        left_root = os.path.join(self.root_dir, self.mode, f'{self.dstype}_left')

        # TODO: verify this glob matches your actual downloaded structure --
        # should return one path per scene folder, e.g. .../clean_left/alley_1
        scene_dirs = sorted(glob(os.path.join(left_root, '*')))
        # print(scene_dirs)
        
        for scene_dir in scene_dirs:
            scene_id = os.path.basename(scene_dir)

            # TODO: glob + sort the frames inside this one scene folder
            left_paths = sorted(glob(os.path.join(scene_dir, '*.png')))
            
            
            # TODO: derive right_paths via .replace(), same pattern as
            # _load_flying -- swap '{dstype}_left' for '{dstype}_right'
            right_paths = [p.replace(f'{self.dstype}_left', f'{self.dstype}_right') for p in left_paths]
            
            
            # TODO: derive disp_paths via .replace() -- swap '{dstype}_left'
            # for 'disparities'. Don't build an occlusions path yourself --
            # readDispSintelStereo does that internally via its own .replace()
            disp_paths = [p.replace(f'{self.dstype}_left', 'disparities') for p in left_paths]
            
            
            # TODO (optional, skip for now): derive flow_paths if you want
            # evaluate_video_scenes.py's flow-warped TEPE to also work here.
            # Remember flow lives under a SEPARATE root (base Sintel download,
            # not sintel_stereo) -- this isn't a same-tree string replace.

            self.scenes.append({
                'scene_id': scene_id,
                'left': left_paths,
                'right': right_paths,
                'disp': disp_paths,
                # 'flow': [],  # placeholder -- fill in later if needed
            })

        self.scenes.sort(key=lambda s: s['scene_id'])

    def _fetch_images(self, image_paths):
        # No changes needed -- identical to SceneFlowVideo._fetch_images,
        # since Sintel's left/right frames are plain RGB PNGs.
        imgs = []
        for img_path in image_paths:
            img = frame_utils.read_gen(img_path)
            img = np.array(img).astype(np.uint8)
            if len(img.shape) == 2:
                img = np.tile(img[..., None], (1, 1, 3))
            else:
                img = img[..., :3]
            img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
            imgs.append(img)
        return imgs

    def _fetch_disparity(self, disp_paths):
        """
        IMPORTANT: must call frame_utils.readDispSintelStereo directly here,
        NOT frame_utils.read_gen -- read_gen just does Image.open() for .png
        and won't decode the R/G/B-packed disparity. This is the crash we
        talked about last message.
        """
        disp_scene, valid_scene = [], []
        for disp_path in disp_paths:
            # TODO: call the reader and convert to the same tensor shapes
            # SceneFlowVideo._fetch_disparity produces:
            #   disp  -> torch.from_numpy(disp).unsqueeze(0)   # (1, H, W)
            #   valid -> torch.from_numpy(valid).float()       # (H, W)
            disp, valid = frame_utils.readDispSintelStereo(disp_path)
            valid = valid & (disp <= self.max_disp) if self.max_disp is not None else valid # filter out any pixels exceeding LAS1's max_disp=192
            
            disp = torch.from_numpy(np.array(disp).astype(np.float32)).unsqueeze(0)  
            valid = torch.from_numpy(np.array(valid).astype(np.float32))
            
            disp_scene.append(disp)
            valid_scene.append(valid)
        return disp_scene, valid_scene

    def __getitem__(self, index):
        scene = self.scenes[index]
        left_imgs = self._fetch_images(scene['left'])
        right_imgs = self._fetch_images(scene['right'])
        disp_scene, valid_scene = self._fetch_disparity(scene['disp'])

        return {
            'scene_id': scene['scene_id'],
            'left': left_imgs,
            'right': right_imgs,
            'disp': disp_scene,
            'valid': valid_scene,
            # 'flow': [],  # TODO: wire up real flow loading if/when you need it
        }
    def __len__(self):
        return len(self.scenes)


class SouthKenSV():
    def __init__(self, pseudo_gt_dir, max_disp=192, border=0):
        self.max_disp = max_disp
        self.border = border
        self.scenes = []

        npz_paths = sorted(glob(osp.join(pseudo_gt_dir, '*.npz')))
        
        for npz_path in sorted(glob(osp.join(pseudo_gt_dir, '*.npz'))):
            with open(npz_path[:-4] + '.json') as f:
                meta = json.load(f)

            self.scenes.append({
                'scene_id': f"{meta['sequence']}",
                'left':     meta['left_paths'],
                'right':    meta['right_paths'],   # add this to the cache script
                'npz':      npz_path,
                'sign':     meta['sign_convention'],
            })

        assert self.scenes, f"No cached sequences found in {pseudo_gt_dir}"

    def _fetch_images(self, image_paths):
            imgs = []
            for img_path in image_paths:
                img = frame_utils.read_gen(img_path)
                img = np.array(img).astype(np.uint8)
                if len(img.shape) == 2:
                    img = np.tile(img[..., None], (1, 1, 3))
                else:
                    img = img[..., :3]
                img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
                imgs.append(img)
            return imgs
        
    def _fetch_disparity(self, npz_path, sign):
        disp = np.load(npz_path)['disparity'].astype(np.float32)   # (T, H, W)
        if sign == 'negative':
            disp = np.abs(disp)

        disp_scene, valid_scene = [], []
        for t in range(disp.shape[0]):
            d = torch.from_numpy(disp[t]).unsqueeze(0)             # (1, H, W)
            valid = (d[0] > 0) & (d[0] <= self.max_disp)

            if self.border > 0:
                b = self.border
                mask = torch.zeros_like(valid)
                mask[b:-b, b:-b] = True
                valid = valid & mask

            disp_scene.append(d)
            valid_scene.append(valid.float())
        return disp_scene, valid_scene
    
    def __getitem__(self, index):
        scene = self.scenes[index]
        disp, valid = self._fetch_disparity(scene['npz'], scene['sign'])
        n = len(disp)

        return {
            'scene_id': scene['scene_id'],
            'left':     self._fetch_images(scene['left'][:n]),
            'right':    self._fetch_images(scene['right'][:n]),
            'disp':     disp,
            'valid':    valid,
        }
    
    def __len__(self):
        return len(self.scenes)   
