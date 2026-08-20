import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
import json
from PIL import Image
from torchvision import transforms
import random
from torchvision.transforms.functional import crop
import numpy as np
from PIL import ImageFile


Image.MAX_IMAGE_PIXELS = 933120000
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CIRRDataset(Dataset):
    def __init__(self, dataset_path, split='test', preprocess=None):
        self.dataset_path = dataset_path
        self.split = split
        self.preprocess = preprocess
        # Load dataset metadata
        with open(os.path.join(dataset_path, f'captions/cap.rc2.{split}.json'), 'r') as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        reference_image_path = os.path.join(self.dataset_path, self.split, item['reference']+'.png')
        reference_image = Image.open(reference_image_path).convert('RGB')
        if self.preprocess:
            reference_image = self.preprocess(reference_image)['pixel_values'][0]

        return {
            'reference_image': reference_image,
            'relative_caption': item['caption'],
            'pairid': item['pairid']
        }


class FashionIQDataset(Dataset):
    def __init__(self, dataset_path, split="val", dress_types=("dress", "shirt", "toptee"), preprocess=None):
        self.dataset_path = dataset_path
        self.split = split
        self.preprocess = preprocess

        self.samples = []
        for dress_type in dress_types:
            cap_file = os.path.join(dataset_path, "captions", f"cap.{dress_type}.{split}.json")

            if not os.path.exists(cap_file):
                continue

            with open(cap_file, "r") as f:
                annotations = json.load(f)

            for idx, ann in enumerate(annotations):
                candidate = ann["candidate"]
                image_path = os.path.join(dataset_path, "images", candidate + ".jpg")

                # bỏ qua nếu thiếu ảnh
                if not os.path.exists(image_path):
                    continue

                self.samples.append(
                    {
                        "pairid": f"{dress_type}_{candidate}_{split}_{idx}",
                        "image_path": image_path,
                        "relative_caption": "; ".join(ann["captions"])
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        if self.preprocess is not None:
            image = self.preprocess(image)["pixel_values"][0]

        return {
            "reference_image": image,
            "relative_caption": sample["relative_caption"],
            "pairid": sample["pairid"],
        }


class CIRCODataset(Dataset):
    def __init__(self, dataset_path, split="val", preprocess=None):
        self.dataset_path = dataset_path
        self.split = split
        self.preprocess = preprocess

        annotation_file = os.path.join(
            dataset_path,
            "annotations",
            f"{split}.json"
        )

        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"CIRCO annotation file not found: {annotation_file}")

        with open(annotation_file, "r") as f:
            self.metadata = json.load(f)

        self.image_dir = os.path.join(
            dataset_path,
            "COCO2017_unlabeled",
            "unlabeled2017"
        )

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"CIRCO image directory not found: {self.image_dir}")

    def __len__(self):
        return len(self.metadata)

    def _get_image_path(self, image_id):
        image_filename = f"{int(image_id):012d}.jpg"
        return os.path.join(self.image_dir, image_filename)

    def __getitem__(self, idx):
        item = self.metadata[idx]

        reference_img_id = item["reference_img_id"]
        reference_image_path = self._get_image_path(reference_img_id)

        if not os.path.exists(reference_image_path):
            raise FileNotFoundError(f"Reference image not found: {reference_image_path}")

        reference_image = Image.open(reference_image_path).convert("RGB")

        if self.preprocess is not None:
            reference_image = self.preprocess(
                reference_image
            )["pixel_values"][0]

        return {
            "reference_image": reference_image,
            "relative_caption": item["relative_caption"],
            "pairid": str(item["id"])
        }

class ComposedEmbedsDataset(Dataset):
    """
    Dataset for loading composed text embeddings.

    It simply scans a directory for *.pt files, making it
    dataset-agnostic (CIRR / FashionIQ / CIRCO / ...).
    """

    def __init__(self, text_embeddings_dir):
        if not os.path.isdir(text_embeddings_dir):
            raise FileNotFoundError(f"Embedding directory not found: {text_embeddings_dir}")
            
        self.text_embeddings_dir = text_embeddings_dir
        self.files = sorted(
            f
            for f in os.listdir(text_embeddings_dir)
            if f.endswith(".pt")
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        pairid = os.path.splitext(filename)[0]

        save_dict = torch.load(
            os.path.join(
                self.text_embeddings_dir,
                filename
            ),
            map_location="cpu"
        )

        cond1 = save_dict["conditioning"]
        cond2 = save_dict["conditioning2"]
        pooled2 = save_dict["pooled2"]

        if cond1.dim() == 3:
            cond1 = cond1.squeeze(0)
        if cond2.dim() == 3:
            cond2 = cond2.squeeze(0)
        if pooled2.dim() == 2:
            pooled2 = pooled2.squeeze(0)

        prompt_embeds = torch.cat([cond1, cond2], dim=-1)

        return {
            "pairid": pairid,
            "prompt_embeds": prompt_embeds,
            "pooled2": pooled2,
        }