import math
import os
from collections import defaultdict

from PIL import Image
from tqdm import tqdm

import torch
from torch import nn, utils
from torchvision import models, datasets, transforms
from utils import *

from .vision import VisionDataset

from pprint import pprint

image_size = [224, 224]
delimiter = '/'

def dict_for_each_episode():
    return [dict() for i in range(18 + 1)]  # episode index: from 1 to 18

def get_model(device):
    print('Loading extractor model: Using ResNet18')

    model = models.resnet18(pretrained=True)
    extractor = nn.Sequential(*list(model.children())[:-2])
    extractor.to(device)

    return extractor

def preprocess_images(args, image_path, cache=True, device=-1, num_workers=40):
    print('Loading Visual')
    visuals = load_visual(args)

    if not (image_path / 'cache').is_dir():
        (image_path / 'cache').mkdir()

    full_image_cache_path  = image_path / 'cache' / 'full_image.pickle'
    person_full_cache_path = image_path / 'cache' / 'person_full.pickle'

    cached = {
        'full_image':  full_image_cache_path.is_file(),
        'person_full': person_full_cache_path.is_file()
    }

    full_images  = dict_for_each_episode()
    person_fulls = dict_for_each_episode()

    if cached['full_image']:
        print("Loading Full Image Feature Cache")
        full_images = load_pickle(full_image_cache_path)

    if cached['person_full']:
        print("Loading Person Full Feature Cache")
        person_fulls = load_pickle(person_full_cache_path)
    
    if not cached['full_image'] or not cached['person_full']:
        print("Loading Image Files and Building Full Image / Person Full Feature Cache")
        
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        model = get_model(device)
        episode_paths = list(image_path.glob('*'))
        for e in tqdm(episode_paths, desc='Episode'):
            shot_paths = list(e.glob('*/*'))  # episode/scene/shot

            # Load image and flatten
            images = load_images(shot_paths)
            images = {"{}{}{}".format(vid, delimiter, name): image for vid, shots in images.items() for name, image in shots.items()}
            dataset = ObjectDataset(args, images, visuals, transform=transform)
            
            full_images_chunk, person_fulls_chunk = extract_features(
                args, dataset, model, cached, device=device, num_workers=num_workers, 
                extractor_batch_size=args.extractor_batch_size
            )

            for total, part in zip(full_images, full_images_chunk):
                total.update(part)

            for total, part in zip(person_fulls, person_fulls_chunk):
                total.update(part)
            
            del images, dataset # delete data to retrieve memory
        del model # delete extractor model to retrieve memory

        if cache:
            if not cached['full_image']:
                print("Saving Full Image Feature Cache As", full_image_cache_path)
                save_pickle(full_images, full_image_cache_path)

            if not cached['person_full']:
                print("Saving Person Full Feature Cache As", person_full_cache_path)
                save_pickle(person_fulls, person_full_cache_path)

    return full_images, person_fulls, visuals

def load_images(shot_paths):
    """
    images = {
        shot1: {
            frame_id1: PIL image1, 
            ...
        }, 
        ...
    }
    """

    images = list(tqdm(map(load_image, shot_paths), total=len(shot_paths), desc='loading images'))
    images = {k: v for k, v in images}

    return images

def load_image(shot_path):
    """
    res = {
        frame_id1: PIL image1, 
        ...
    }
    """

    image_paths = shot_path.glob('*')
    vid = '_'.join(shot_path.parts[-3:])
    res = {}
    image_paths = sorted(list(image_paths))
    for image_path in image_paths:
        name = image_path.parts[-1] # name ex) IMAGE_0000046147.jpg
        image = Image.open(image_path)
        res[name] = image

    return (vid, res)

def load_visual(args):
    visual = load_json(args.visual_path)
    visual_by_episode = dict_for_each_episode()

    for shot, frames in visual.items():
        episode = get_episode_id(shot)
        episode_dict = visual_by_episode[episode]

        for frame in frames:
            frame_id = get_frame_id(frame['frame_id']) 
            episode_dict[frame_id] = frame

    return visual_by_episode

class ObjectDataset(VisionDataset):
    def __init__(self, args, images, visuals, **kwargs):
        super(ObjectDataset, self).__init__('~/', **kwargs)

        self.args = args
        self.images = list([(k, v) for k, v in images.items()])
        self.visuals = visuals

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        key, pil_full_img = self.images[idx]

        episode = get_episode_id(key)
        frame = get_frame_id(key)

        if self.transform is not None:
            full_img = self.transform(pil_full_img)  

        person_fulls = []
        visuals = self.visuals[episode]
        if frame in visuals:
            persons = visuals[frame]["persons"]

            for p in persons:
                full_rect = p["person_info"]["full_rect"]

                if full_rect["max_x"] != '':
                    person_full = transforms.functional.crop(pil_full_img, *self.bbox_transform(full_rect))
                    if self.transform is not None:
                        person_full = self.transform(person_full)
                else: # no bounding box: use full image
                    person_full = full_img

                person_fulls.append(person_full)

        if not person_fulls: # empty (no visual data or no person) - use full image
            person_fulls.append(full_img)

        person_fulls = torch.stack(person_fulls)

        return (episode, frame), full_img, person_fulls

    def collate_fn(self, batch):
        keys, full_imgs, person_fulls = zip(*batch)

        full_imgs = torch.stack(full_imgs)
        person_fulls = list(person_fulls)

        return keys, full_imgs, person_fulls

    def bbox_transform(self, rect):
        """min_x, min_y, max_x, max_y -> top left corner coordinates, height, width"""

        top_left_v = rect["min_y"]
        top_left_h = rect["min_x"]
        height = rect["max_y"] - top_left_v
        width = rect["max_x"] - top_left_h

        return top_left_v, top_left_h, height, width


def mean_pool(tensor, dim):
    return torch.mean(tensor, dim=dim, keepdim=False)

def extract_and_pool(tensor, model, device):
    tensor = tensor.to(device)
    tensor = model(tensor)          # N X C x H x W (N: extractor_batch_size / number of person fulls in a frame, C: 512)
    tensor = mean_pool(tensor, -1)  # N X C x H 
    tensor = mean_pool(tensor, -1)  # N X C
    tensor = tensor.cpu().numpy()

    return tensor

def extract_features(args, dataset, model, cached, device=-1, num_workers=1, extractor_batch_size=384):
    """
    full_images_by_episode = [
        {}, # empty dict

        { (episode1)
            frame_id: vector, # shape: (C,)
            ...
        },

        ...

        { (episode18)
            frame_id: vector, 
            ...
        }
    ]

    person_fulls_by_episode = [
        {}, # empty dict

        { (episode1)
            frame_id: matrix, # shape: (N, C) N: number of person
            ...
        },

        ...

        { (episode18)
            frame_id: matrix, 
            ...
        }
    ]
    """

    dataloader = utils.data.DataLoader(
        dataset,
        batch_size=extractor_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn
    )

    model.eval()
    full_images_by_episode  = dict_for_each_episode()
    person_fulls_by_episode = dict_for_each_episode()
    with torch.no_grad():
        for data in tqdm(dataloader, total=math.ceil(len(dataset) / extractor_batch_size), desc='extracting features'):
            keys, full_imgs, person_fulls = data

            if not cached['full_image']:
                full_imgs = extract_and_pool(full_imgs, model, device) 

                for (e, f), fi, in zip(keys, full_imgs):
                    full_images_by_episode[e][f] = fi

            if not cached['person_full']:
                person_fulls = [extract_and_pool(pfu, model, device) for pfu in person_fulls]

                for (e, f), pfu in zip(keys, person_fulls):
                    person_fulls_by_episode[e][f] = pfu
    
    del dataloader

    return full_images_by_episode, person_fulls_by_episode



