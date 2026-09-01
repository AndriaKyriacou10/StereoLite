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

class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None, real_world=False):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []
        self.real_world = real_world

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        try:
            disp = self.disparity_reader(self.disparity_list[index])
            if isinstance(disp, tuple):
                disp, valid = disp
            else:
                valid = disp < 192

            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])

            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)

            disp = np.array(disp).astype(np.float32)

            flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

            # grayscale images
            if len(img1.shape) == 2:
                img1 = np.tile(img1[..., None], (1, 1, 3))
            else:
                img1 = img1[..., :3]

            if len(img2.shape) == 2:
                img2 = np.tile(img2[..., None], (1, 1, 3))
            else:
                img2 = img2[..., :3]

            if self.augmentor is not None:
                if self.sparse:
                    img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
                else:
                    img1, img2, flow = self.augmentor(img1, img2, flow)

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            
            if self.sparse:
                valid = torch.from_numpy(valid)
            else:
                valid = (flow[0].abs() < 192) & (flow[1].abs() < 192)

            if self.img_pad is not None:
                padH, padW = self.img_pad
                img1 = F.pad(img1, [padW] * 2 + [padH] * 2)
                img2 = F.pad(img2, [padW] * 2 + [padH] * 2)

            flow = flow[:1]
            return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()

        except Exception as e:
            # Useful for locating the file later:
            print(f"[WARN] Skipping corrupted sample:\n  {self.image_list[index]}\n  Err={e}")
            return self.__getitem__(0)

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self

    def __len__(self):
        return len(self.image_list)


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='./data/datasets/ETH3D', split='training'):
        super(ETH3D, self).__init__(aug_params, sparse=True)

        image1_list = sorted(glob(osp.join(root, f'two_view_{split}/*/im0.png')))
        image2_list = sorted(glob(osp.join(root, f'two_view_{split}/*/im1.png')))
        disp_list = sorted(glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm'))) \
            if split == 'training' \
            else [osp.join(root,'two_view_training_gt/playground_1l/disp0GT.pfm')] * len(image1_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]



class KITTI(StereoDataset):
    def __init__(self, aug_params=None, image_set='training', year=2015):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        if year == 2012:
            root_12 = './data/datasets/kitti12'
            image1_list = sorted(glob(os.path.join(root_12, image_set, 'colored_0/*_10.png')))
            image2_list = sorted(glob(os.path.join(root_12, image_set, 'colored_1/*_10.png')))
            disp_list = sorted(
                glob(os.path.join(root_12, image_set, 'disp_occ/*_10.png')))

        if year == 2015:
            root_15 = './data/datasets/kitti15'
            image1_list = sorted(glob(os.path.join(root_15, image_set, 'image_2/*_10.png')))
            image2_list = sorted(glob(os.path.join(root_15, image_set, 'image_3/*_10.png')))
            disp_list = sorted(
                glob(os.path.join(root_15, image_set, 'disp_occ_0/*_10.png')))

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='./data/datasets/Middlebury', split='2014', resolution='F'):
        super(Middlebury, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispMiddlebury)
        assert os.path.exists(root)
        assert split in ["2005", "2006", "2014", "2021", "MiddEval3"]
        if split == "2005":
            scenes = list((Path(root) / "2005").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "view1.png"), str(scene / "view5.png")]]
                self.disparity_list += [str(scene / "disp1.png")]
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:
                        self.image_list += [[str(scene / f"Illum{illum}/Exp{exp}/view1.png"),
                                             str(scene / f"Illum{illum}/Exp{exp}/view5.png")]]
                        self.disparity_list += [str(scene / "disp1.png")]
        elif split == "2006":
            scenes = list((Path(root) / "2006").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "view1.png"), str(scene / "view5.png")]]
                self.disparity_list += [str(scene / "disp1.png")]
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:
                        self.image_list += [[str(scene / f"Illum{illum}/Exp{exp}/view1.png"),
                                             str(scene / f"Illum{illum}/Exp{exp}/view5.png")]]
                        self.disparity_list += [str(scene / "disp1.png")]
        elif split == "2014":
            scenes = list((Path(root) / "2014").glob("*"))
            for scene in scenes:
                for s in ["E", "L", ""]:
                    self.image_list += [[str(scene / "im0.png"), str(scene / f"im1{s}.png")]]
                    self.disparity_list += [str(scene / "disp0.pfm")]
        elif split == "2021":
            scenes = list((Path(root) / "2021/data").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "im0.png"), str(scene / "im1.png")]]
                self.disparity_list += [str(scene / "disp0.pfm")]
                for s in ["0", "1", "2", "3"]:
                    if os.path.exists(str(scene / f"ambient/L0/im0e{s}.png")):
                        self.image_list += [
                            [str(scene / f"ambient/L0/im0e{s}.png"), str(scene / f"ambient/L0/im1e{s}.png")]]
                        self.disparity_list += [str(scene / "disp0.pfm")]
        else:
            image1_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/im0.png')))
            image2_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/im1.png')))
            disp_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/disp0GT.pfm')))
            assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]


class DrivingStereoWeather(StereoDataset):
    def __init__(self, aug_params=None, root='./data/datasets/DrivingStereoWeather', image_set='rainy'):
        super(DrivingStereoWeather, self).__init__(aug_params, sparse=True,
                                                   reader=frame_utils.readDispDrivingStereoFull)
        assert os.path.exists(root), f"DrivingStereoWeather root not found: {root}"

        image1_list = sorted(glob(os.path.join(root, image_set, 'left-image-full-size/*.png')))
        image2_list = sorted(glob(os.path.join(root, image_set, 'right-image-full-size/*.png')))
        disp_list = sorted(glob(os.path.join(root, image_set, 'disparity-map-full-size/*.png')))
        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]

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
