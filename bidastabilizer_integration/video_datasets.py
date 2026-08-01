# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# # Data loading based on https://github.com/NVIDIA/flownet2-pytorch


import os
import copy
import gzip
import logging
import torch
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
import os.path as osp
from glob import glob
import cv2
import re

from collections import defaultdict
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional
# from pytorch3d.renderer.cameras import PerspectiveCameras
# from pytorch3d.implicitron.dataset.types import (
#     FrameAnnotation as ImplicitronFrameAnnotation,
#     load_dataclass,
# )

from core.utils import frame_utils
# from bidavideo.evaluation.utils.eval_utils import depth2disparity_scale
from bidastabilizer_integration.augmentor import SequenceDispFlowAugmentor, SequenceDispSparseFlowAugmentor


class StereoSequenceDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = (
            aug_params.pop("img_pad", None) if aug_params is not None else None
        )
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SequenceDispSparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = SequenceDispFlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader
        self.depth_reader = self._load_depth
        self.is_test = False
        self.sample_list = []
        self.extra_info = []
        self.depth_eps = 1e-5

    def _load_depth(self, depth_path):
        if depth_path[-3:] == "npy":
            return self._load_npy_depth(depth_path)
        elif depth_path[-3:] == "png":
            if "kitti" in depth_path:
                return self._load_kitti_depth(depth_path)
            else:
                return self._load_16big_png_depth(depth_path)
        else:
            raise ValueError("Other format depth is not implemented")

    def _load_npy_depth(self, depth_npy):
        depth = np.load(depth_npy)
        return depth

    def _get_output_tensor(self, sample):
        output_tensor = defaultdict(list)
        sample_size = len(sample["image"]["left"])
        output_tensor_keys = ["img", "disp", "valid_disp", "mask"]
        add_keys = ["viewpoint", "metadata"]
        for add_key in add_keys:
            if add_key in sample:
                output_tensor_keys.append(add_key)

        for key in output_tensor_keys:
            output_tensor[key] = [[] for _ in range(sample_size)]

        if "viewpoint" in sample:
            viewpoint_left = self._get_pytorch3d_camera(
                sample["viewpoint"]["left"][0],
                sample["metadata"]["left"][0][1],
                scale=1.0,
            )
            viewpoint_right = self._get_pytorch3d_camera(
                sample["viewpoint"]["right"][0],
                sample["metadata"]["right"][0][1],
                scale=1.0,
            )
            depth2disp_scale = depth2disparity_scale(
                viewpoint_left,
                viewpoint_right,
                torch.Tensor(sample["metadata"]["left"][0][1])[None],
            )
            output_tensor["depth2disp_scale"] = [depth2disp_scale]

        if "camera" in sample:
            output_tensor["viewpoint"] = [[] for _ in range(sample_size)]

            # InfinigenSV
            if sample["camera"]["left"][0][-3:] == "npz":
                # Note that the K, R, T is based on Blender world Matrix
                camera_left = np.load(sample["camera"]["left"][0])
                camera_right = np.load(sample["camera"]["right"][0])
                camera_left_RT = camera_left['T']
                camera_right_RT = camera_right['T']
                camera_left_K = camera_left['K']
                camera_right_K = camera_right['K']
                camera_left_T = camera_left['T'][:3, 3]
                camera_left_R = camera_left['T'][:3, :3]
                fix_baseline = np.linalg.norm(camera_left_RT[:3, 3] - camera_right_RT[:3, 3])
                focal_length_px = camera_left_K[0][0]
                depth2disp_scale = focal_length_px * fix_baseline

            # Sintel
            elif sample["camera"]["left"][0][-3:] == "cam":
                TAG_FLOAT = 202021.25
                f = open(sample["camera"]["left"][0], 'rb')
                check = np.fromfile(f, dtype=np.float32, count=1)[0]
                assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(
                    TAG_FLOAT, check)
                camera_left_K = np.fromfile(f, dtype='float64', count=9).reshape((3, 3))
                camera_left_RT = np.fromfile(f, dtype='float64', count=12).reshape((3, 4))
                fix_baseline = 0.1 # From the MPI Sintel dataset website, the baseline of the cameras = 10cm = 0.1m
                focal_length_px = camera_left_K[0][0]
                depth2disp_scale = focal_length_px * fix_baseline
                camera_left_T = camera_left_RT[:3, 3]
                camera_left_R = camera_left_RT[:3, :3]

            # KITTI Depth
            elif sample["camera"]["left"][0][-20:] == "calib_cam_to_cam.txt":
                calib_data = {}
                with open(sample["camera"]["left"][0], 'r') as f:
                    for line in f:
                        key, value = line.split(':', 1)
                        calib_data[key.strip()] = value.strip()

                P_key = 'P_rect_02'
                if P_key in calib_data:
                    P_values = np.array([float(x) for x in calib_data[P_key].split()])
                    P_matrix = P_values.reshape(3, 4)
                else:
                    raise KeyError(f"Projection matrix for camera not found in calibration data")
                focal_length_px = P_matrix[0, 0]

                T_key1 = 'T_02'
                T_key2 = 'T_03'
                if T_key1 in calib_data and T_key2 in calib_data:
                    T1 = np.array([float(x) for x in calib_data[T_key1].split()])
                    T2 = np.array([float(x) for x in calib_data[T_key2].split()])
                    baseline = np.linalg.norm(T1 - T2)
                else:
                    raise KeyError(f"Translation vectors for cameras not found in calibration data")

                R_key1 = 'R_rect_02'
                R_key2 = 'R_rect_03'
                if R_key1 in calib_data and R_key2 in calib_data:
                    R1 = np.array([float(x) for x in calib_data[R_key1].split()]).reshape(3, 3)
                    R2 = np.array([float(x) for x in calib_data[R_key2].split()]).reshape(3, 3)
                else:
                    raise KeyError(f"Rotation vectors for cameras not found in calibration data")

                depth2disp_scale = focal_length_px * baseline
                camera_left_K = P_matrix[:, :3]
                camera_left_T = T1
                camera_left_R = R1

            # SouthKensington
            elif sample["camera"]["left"][0][-3:] == "txt":
                camera_left_K, frames = self.parse_txt_file(sample["camera"]["left"][0])
                fix_baseline = 0.12
                camera_left_R = frames[0]['R']
                camera_left_T = frames[0]['T']
                focal_length_px = camera_left_K[0][0]
                depth2disp_scale = focal_length_px * fix_baseline
            else:
                raise ValueError("Other format camera is not implemented")

            output_tensor["depth2disp_scale"] = [depth2disp_scale]
            output_tensor["RTK"] = [camera_left_R, camera_left_T, camera_left_K]

        for i in range(sample_size):
            for cam in ["left", "right"]:
                if "mask" in sample and cam in sample["mask"]:
                    mask = frame_utils.read_gen(sample["mask"][cam][i])
                    mask = np.array(mask) / 255.0
                    output_tensor["mask"][i].append(mask)

                if "viewpoint" in sample and cam in sample["viewpoint"]:
                    viewpoint = self._get_pytorch3d_camera(
                        sample["viewpoint"][cam][i],
                        sample["metadata"][cam][i][1],
                        scale=1.0,
                    )
                    output_tensor["viewpoint"][i].append(viewpoint)
                if "camera" in sample:
                    # InfinigenSV
                    if sample["camera"]["left"][0][-3:] == "npz" and cam in sample["camera"]:
                        # Note that the K, R, T is based on Blender world Matrix
                        camera = np.load(sample["camera"][cam][i])
                        camera_K = camera['K']
                        camera_T = camera['T'][:3, 3]
                        camera_R = camera['T'][:3, :3]
                        viewpoint = self._get_pytorch3d_camera_from_blender(
                            camera_R, camera_T, camera_K,
                            sample["metadata"][cam][i][1],
                            scale=1.0,
                        )
                        output_tensor["viewpoint"][i].append(viewpoint)

                    # Sintel
                    elif sample["camera"]["left"][0][-3:] == "cam" and cam in sample["camera"]:
                        f = open(sample["camera"]["left"][0], 'rb')
                        check = np.fromfile(f, dtype=np.float32, count=1)[0]
                        assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(
                            TAG_FLOAT, check)
                        camera_K = np.fromfile(f, dtype='float64', count=9).reshape((3, 3))
                        camera_RT = np.fromfile(f, dtype='float64', count=12).reshape((3, 4))
                        camera_T = camera_RT[:3, 3]
                        camera_R = camera_RT[:3, :3]
                        viewpoint = self._get_pytorch3d_camera_from_blender(
                            camera_R, camera_T, camera_K,
                            sample["metadata"][cam][i][1],
                            scale=1.0,
                        )
                        output_tensor["viewpoint"][i].append(viewpoint)

                    # KITTI Depth
                    elif sample["camera"]["left"][0][-20:] == "calib_cam_to_cam.txt":
                        calib_data = {}
                        with open(sample["camera"]["left"][0], 'r') as f:
                            for line in f:
                                key, value = line.split(':', 1)
                                calib_data[key.strip()] = value.strip()

                        P_key = 'P_rect_02'
                        if P_key in calib_data:
                            P_values = np.array([float(x) for x in calib_data[P_key].split()])
                            P_matrix = P_values.reshape(3, 4)
                        else:
                            raise KeyError(f"Projection matrix for camera not found in calibration data")
                        focal_length_px = P_matrix[0, 0]

                        T_key1 = 'T_02'
                        T_key2 = 'T_03'
                        if T_key1 in calib_data and T_key2 in calib_data:
                            T1 = np.array([float(x) for x in calib_data[T_key1].split()])
                            T2 = np.array([float(x) for x in calib_data[T_key2].split()])
                            baseline = np.linalg.norm(T1 - T2)
                        else:
                            raise KeyError(f"Translation vectors for cameras not found in calibration data")

                        R_key1 = 'R_rect_02'
                        R_key2 = 'R_rect_03'
                        if R_key1 in calib_data and R_key2 in calib_data:
                            R1 = np.array([float(x) for x in calib_data[R_key1].split()]).reshape(3, 3)
                            R2 = np.array([float(x) for x in calib_data[R_key2].split()]).reshape(3, 3)
                        else:
                            raise KeyError(f"Rotation vectors for cameras not found in calibration data")

                        depth2disp_scale = focal_length_px * baseline
                        camera_K = P_matrix[:, :3]
                        camera_T = T1
                        camera_R = R1

                        viewpoint = self._get_pytorch3d_camera_from_blender(
                            camera_R, camera_T, camera_K,
                            sample["metadata"][cam][i][1],
                            scale=1.0,
                        )
                        output_tensor["viewpoint"][i].append(viewpoint)

                    # SouthKensington
                    elif sample["camera"]["left"][0][-3:] == "txt" and cam in sample["camera"]:
                        camera_left_K, frames = self.parse_txt_file(sample["camera"]["left"][0])

                        camera_K = camera_left_K
                        camera_R = frames[i]['R']
                        camera_T = frames[i]['T']
                        viewpoint = self._get_pytorch3d_camera_from_blender(
                            camera_R, camera_T, camera_K,
                            sample["metadata"][cam][i][1],
                            scale=1.0,
                        )
                        output_tensor["viewpoint"][i].append(viewpoint)

                if "metadata" in sample and cam in sample["metadata"]:
                    metadata = sample["metadata"][cam][i]
                    output_tensor["metadata"][i].append(metadata)

                if cam in sample["image"]:
                    img = frame_utils.read_gen(sample["image"][cam][i])
                    img = np.array(img).astype(np.uint8)

                    # grayscale images
                    if len(img.shape) == 2:
                        img = np.tile(img[..., None], (1, 1, 3))
                    else:
                        img = img[..., :3]
                    output_tensor["img"][i].append(img)

                if cam in sample["disparity"]:
                    disp = self.disparity_reader(sample["disparity"][cam][i])
                    if isinstance(disp, tuple):
                        disp, valid_disp = disp
                    else:
                        valid_disp = disp < 192
                    disp = np.array(disp).astype(np.float32)

                    disp = np.stack([-disp, np.zeros_like(disp)], axis=-1)

                    output_tensor["disp"][i].append(disp)
                    output_tensor["valid_disp"][i].append(valid_disp)

                elif "depth" in sample and cam in sample["depth"]:
                    depth = self.depth_reader(sample["depth"][cam][i])
                    depth_mask = depth < self.depth_eps
                    depth[depth_mask] = self.depth_eps
                    disp = depth2disp_scale / depth
                    disp[depth_mask] = 0
                    valid_disp = (disp < 192) * (1 - depth_mask)
                    disp = np.array(disp).astype(np.float32)
                    disp = np.stack([-disp, np.zeros_like(disp)], axis=-1)
                    output_tensor["disp"][i].append(disp)
                    output_tensor["valid_disp"][i].append(valid_disp)

        return output_tensor

    def __getitem__(self, index):
        im_tensor = {"img"}
        sample = self.sample_list[index]
        if self.is_test:
            sample_size = len(sample["image"]["left"])
            im_tensor["img"] = [[] for _ in range(sample_size)]
            for i in range(sample_size):
                for cam in ["left", "right"]:
                    img = frame_utils.read_gen(sample["image"][cam][i])
                    img = np.array(img).astype(np.uint8)[..., :3]
                    img = torch.from_numpy(img).permute(2, 0, 1).float()
                    im_tensor["img"][i].append(img)
            im_tensor["img"] = torch.stack(im_tensor["img"])
            return im_tensor, self.extra_info[index]

        index = index % len(self.sample_list)
        try:
            output_tensor = self._get_output_tensor(sample)
        except:
            logging.warning(f"Exception in loading sample {index}!")
            index = np.random.randint(len(self.sample_list))
            logging.info(f"New index is {index}")
            sample = self.sample_list[index]
            output_tensor = self._get_output_tensor(sample)

        sample_size = len(sample["image"]["left"])
        if self.augmentor is not None:
            if self.sparse:
                output_tensor["img"], output_tensor["disp"], output_tensor["valid_disp"] = self.augmentor(
                    output_tensor["img"], output_tensor["disp"], output_tensor["valid_disp"]
                )
            else:
                output_tensor["img"], output_tensor["disp"] = self.augmentor(
                    output_tensor["img"], output_tensor["disp"]
                )
        for i in range(sample_size):
            for cam in (0, 1):
                if cam < len(output_tensor["img"][i]):
                    img = (
                        torch.from_numpy(output_tensor["img"][i][cam])
                        .permute(2, 0, 1)
                        .float()
                    )
                    if self.img_pad is not None:
                        padH, padW = self.img_pad
                        img = F.pad(img, [padW] * 2 + [padH] * 2)
                    output_tensor["img"][i][cam] = img

                if cam < len(output_tensor["disp"][i]):
                    disp = (
                        torch.from_numpy(output_tensor["disp"][i][cam])
                        .permute(2, 0, 1)
                        .float()
                    )

                    if self.sparse:
                        valid_disp = torch.from_numpy(
                            output_tensor["valid_disp"][i][cam]
                        )
                    else:
                        valid_disp = (
                            (disp[0].abs() < 192)
                            & (disp[1].abs() < 192)
                            & (disp[0].abs() != 0)
                        )
                    disp = disp[:1]

                    output_tensor["disp"][i][cam] = disp
                    output_tensor["valid_disp"][i][cam] = valid_disp.float()

                if "mask" in output_tensor and cam < len(output_tensor["mask"][i]):
                    mask = torch.from_numpy(output_tensor["mask"][i][cam]).float()
                    output_tensor["mask"][i][cam] = mask

                if "viewpoint" in output_tensor and cam < len(
                    output_tensor["viewpoint"][i]
                ):

                    viewpoint = output_tensor["viewpoint"][i][cam]
                    output_tensor["viewpoint"][i][cam] = viewpoint

        res = {}
        if "viewpoint" in output_tensor and self.split != "train":
            res["viewpoint"] = output_tensor["viewpoint"]
        if "metadata" in output_tensor and self.split != "train":
            res["metadata"] = output_tensor["metadata"]
        if "depth2disp_scale" in output_tensor and self.split != "train":
            res["depth2disp_scale"] = output_tensor["depth2disp_scale"]
        if "RTK" in output_tensor and self.split != "train":
            res["RTK"] = output_tensor["RTK"]

        for k, v in output_tensor.items():
            if k != "viewpoint" and k != "metadata" and k != "depth2disp_scale" and k != "RTK":
                for i in range(len(v)):
                    if len(v[i]) > 0:
                        v[i] = torch.stack(v[i])
                if len(v) > 0 and (len(v[0]) > 0):
                    res[k] = torch.stack(v)
        return res

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.sample_list = v * copy_of_self.sample_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self

    def __len__(self):
        return len(self.sample_list)

class SequenceSceneFlowDataset(StereoSequenceDataset):
    def __init__(
        self,
        aug_params=None,
        root=os.path.expanduser("~/IRP/data/datasets/SceneFlow"),
        dstype="frames_cleanpass",
        sample_len=1,
        things_test=False,
        add_things=True,
        add_monkaa=True,
        add_driving=True,
    ):
        super(SequenceSceneFlowDataset, self).__init__(aug_params)
        self.root = root
        self.dstype = dstype
        self.sample_len = sample_len
        if things_test:
            self._add_things("TEST")
        else:
            if add_things:
                self._add_things("TRAIN")
            if add_monkaa:
                self._add_monkaa()
            if add_driving:
                self._add_driving()

    def _add_things(self, split="TRAIN"):
        """Add FlyingThings3D data"""

        original_length = len(self.sample_list)
        root = osp.join(self.root, "FlyingThings3D")
        image_paths = defaultdict(list)
        disparity_paths = defaultdict(list)
        print(os.path.exists(root))
        for cam in ["left", "right"]:
            image_paths[cam] = sorted(
                glob(osp.join(root, self.dstype, split, f"*/*/{cam}/"))
            )
            disparity_paths[cam] = [
                path.replace(self.dstype, "disparity") for path in image_paths[cam]
            ]
        # Choose a random subset of 400 images for validation
        # state = np.random.get_state()
        # np.random.seed(1000)
        # val_idxs = set(np.random.permutation(len(image_paths["left"]))[:40])
        # np.random.set_state(state)
        # np.random.seed(0)
        num_seq = len(image_paths["left"])
        num = 0
        for seq_idx in range(num_seq):
            # if (split == "TEST" and seq_idx in val_idxs) or (
            #     split == "TRAIN" and not seq_idx in val_idxs
            # ):
            images, disparities = defaultdict(list), defaultdict(list)
            for cam in ["left", "right"]:
                images[cam] = sorted(
                    glob(osp.join(image_paths[cam][seq_idx], "*.png"))
                )
                disparities[cam] = sorted(
                    glob(osp.join(disparity_paths[cam][seq_idx], "*.pfm"))
                )
            num = num + len(images["left"])
            self._append_sample(images, disparities)
        print(num)
        assert len(self.sample_list) > 0, "No samples found"
        print(
            f"Added {len(self.sample_list) - original_length} from FlyingThings {self.dstype}"
        )
        logging.info(
            f"Added {len(self.sample_list) - original_length} from FlyingThings {self.dstype}"
        )

    def _add_monkaa(self):
        """Add FlyingThings3D data"""

        original_length = len(self.sample_list)
        root = osp.join(self.root, "Monkaa")
        image_paths = defaultdict(list)
        disparity_paths = defaultdict(list)

        for cam in ["left", "right"]:
            image_paths[cam] = sorted(glob(osp.join(root, self.dstype, f"*/{cam}/")))
            disparity_paths[cam] = [
                path.replace(self.dstype, "disparity") for path in image_paths[cam]
            ]

        num_seq = len(image_paths["left"])

        for seq_idx in range(num_seq):
            images, disparities = defaultdict(list), defaultdict(list)
            for cam in ["left", "right"]:
                images[cam] = sorted(glob(osp.join(image_paths[cam][seq_idx], "*.png")))
                disparities[cam] = sorted(
                    glob(osp.join(disparity_paths[cam][seq_idx], "*.pfm"))
                )

            self._append_sample(images, disparities)

        assert len(self.sample_list) > 0, "No samples found"
        print(
            f"Added {len(self.sample_list) - original_length} from Monkaa {self.dstype}"
        )
        logging.info(
            f"Added {len(self.sample_list) - original_length} from Monkaa {self.dstype}"
        )

    def _add_driving(self):
        """Add FlyingThings3D data"""

        original_length = len(self.sample_list)
        root = osp.join(self.root, "Driving")
        image_paths = defaultdict(list)
        disparity_paths = defaultdict(list)

        for cam in ["left", "right"]:
            image_paths[cam] = sorted(
                glob(osp.join(root, self.dstype, f"*/*/*/{cam}/"))
            )
            disparity_paths[cam] = [
                path.replace(self.dstype, "disparity") for path in image_paths[cam]
            ]

        num_seq = len(image_paths["left"])
        for seq_idx in range(num_seq):
            images, disparities = defaultdict(list), defaultdict(list)
            for cam in ["left", "right"]:
                images[cam] = sorted(glob(osp.join(image_paths[cam][seq_idx], "*.png")))
                disparities[cam] = sorted(
                    glob(osp.join(disparity_paths[cam][seq_idx], "*.pfm"))
                )

            self._append_sample(images, disparities)

        assert len(self.sample_list) > 0, "No samples found"
        print(
            f"Added {len(self.sample_list) - original_length} from Driving {self.dstype}"
        )
        logging.info(
            f"Added {len(self.sample_list) - original_length} from Driving {self.dstype}"
        )

    def _append_sample(self, images, disparities):
        seq_len = len(images["left"])
        for ref_idx in range(0, seq_len - self.sample_len):
            sample = defaultdict(lambda: defaultdict(list))
            for cam in ["left", "right"]:
                for idx in range(ref_idx, ref_idx + self.sample_len):
                    sample["image"][cam].append(images[cam][idx])
                    sample["disparity"][cam].append(disparities[cam][idx])
            self.sample_list.append(sample)

            sample = defaultdict(lambda: defaultdict(list))
            for cam in ["left", "right"]:
                for idx in range(ref_idx, ref_idx + self.sample_len):
                    sample["image"][cam].append(images[cam][seq_len - idx - 1])
                    sample["disparity"][cam].append(disparities[cam][seq_len - idx - 1])
            self.sample_list.append(sample)

class SequenceSintelStereo(StereoSequenceDataset):
    def __init__(
        self,
        dstype="clean",
        aug_params=None,
        root="./data/datasets",
    ):
        super().__init__(
            aug_params, sparse=True, reader=frame_utils.readDispSintelStereo
        )
        self.dstype = dstype
        self.split = "test"
        original_length = len(self.sample_list)
        image_root = osp.join(root, "sintel_stereo", "training")
        image_paths = defaultdict(list)
        disparity_paths = defaultdict(list)
        camera_paths = defaultdict(list)

        for cam in ["left", "right"]:
            image_paths[cam] = sorted(
                glob(osp.join(image_root, f"{self.dstype}_{cam}/*"))
            )

        cam = "left"
        disparity_paths[cam] = [
            path.replace(f"{self.dstype}_{cam}", "disparities")
            for path in image_paths[cam]
        ]
        camera_paths[cam] = [
            path.replace(f"{self.dstype}_{cam}", "camdata_left")
            for path in image_paths[cam]
        ]

        num_seq = len(image_paths["left"])
        # for each sequence
        for seq_idx in range(num_seq):
            sequence_name = os.path.basename(image_paths[cam][seq_idx])
            sample = defaultdict(lambda: defaultdict(list))
            for cam in ["left", "right"]:
                sample["image"][cam] = sorted(
                    glob(osp.join(image_paths[cam][seq_idx], "*.png"))
                )
                for _ in range(len(sample["image"][cam])):
                    sample["metadata"][cam].append([sequence_name, (436, 1024)])

            cam = "left"
            sample["disparity"][cam] = sorted(
                glob(osp.join(disparity_paths[cam][seq_idx], "*.png"))
            )
            sample["camera"][cam] = sorted(
                glob(osp.join(camera_paths[cam][seq_idx], "*.cam"))
            )

            for im1, disp, camera in zip(sample["image"][cam], sample["disparity"][cam], sample["camera"][cam]):
                assert (
                    im1.split("/")[-1].split(".")[0]
                    == disp.split("/")[-1].split(".")[0]
                    == camera.split("/")[-1].split(".")[0]
                ), (im1.split("/")[-1].split(".")[0], disp.split("/")[-1].split(".")[0], camera.split("/")[-1].split(".")[0])
            self.sample_list.append(sample)

        logging.info(
            f"Added {len(self.sample_list) - original_length} from SintelStereo {self.dstype}"
        )


def fetch_dataloader(args):
    """Create the data loader for the corresponding training set"""

    aug_params = {
        "crop_size": args.image_size,
        "min_scale": args.spatial_scale[0],
        "max_scale": args.spatial_scale[1],
        "do_flip": False,
        "yjitter": not args.noyjitter,
    }
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None

    add_monkaa = "monkaa" in args.train_datasets
    add_driving = "driving" in args.train_datasets
    add_things = "things" in args.train_datasets
    add_dynamic_replica = "dynamic_replica" in args.train_datasets
    add_infinigensv = "infinigen_stereovideo" in args.train_datasets
    add_kittidepth = "kitti_depth" in args.train_datasets
    new_dataset = None

    if add_monkaa or add_driving or add_things:
        # clean_dataset = SequenceSceneFlowDataset(
        #     aug_params,
        #     dstype="frames_cleanpass",
        #     sample_len=args.sample_len,
        #     add_monkaa=add_monkaa,
        #     add_driving=add_driving,
        #     add_things=add_things,
        # )

        final_dataset = SequenceSceneFlowDataset(
            aug_params,
            dstype="frames_cleanpass",
            sample_len=args.sample_len,
            add_monkaa=add_monkaa,
            add_driving=add_driving,
            add_things=add_things,
        )
        # new_dataset = clean_dataset + final_dataset

        new_dataset = final_dataset

    if add_dynamic_replica:
        dr_dataset = DynamicReplicaDataset(
            aug_params, split="train", sample_len=args.sample_len
        )
        if new_dataset is None:
            new_dataset = dr_dataset
        else:
            new_dataset = new_dataset + dr_dataset

    if add_infinigensv:
        infinigensv_dataset = InfinigenStereoVideoDataset(
            aug_params, split="train", sample_len=args.sample_len
        )
        if new_dataset is None:
            new_dataset = infinigensv_dataset
        else:
            new_dataset = new_dataset + infinigensv_dataset + infinigensv_dataset + infinigensv_dataset

    if add_kittidepth:
        kittidepth_dataset = KITTIDepthDataset(
            aug_params, split="train", sample_len=args.sample_len
        )
        if new_dataset is None:
            new_dataset = kittidepth_dataset
        else:
            new_dataset = new_dataset + kittidepth_dataset

    logging.info(f"Adding {len(new_dataset)} samples in total")
    train_dataset = (
        new_dataset if train_dataset is None else train_dataset + new_dataset
    )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    logging.info("Training with %d image pairs" % len(train_dataset))
    return train_loader
