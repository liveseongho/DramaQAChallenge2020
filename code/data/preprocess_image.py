import math
import os
from collections import defaultdict
# from multiprocessing import Pool

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


def preprocess_images(args, image_path, cache=True, to_features=True, device=-1, num_workers=40):
    cache_path = image_path / 'cache' / 'image.pickle'
    cache_by_episode_path = image_path / 'cache' / 'image_by_episode.pickle'
    cache_by_region_path = image_path / 'cache' / 'region_by_episode.pickle'

    if not (image_path / 'cache').is_dir():
        (image_path / 'cache').mkdir()

    if not to_features:
        raise ValueError('to_features=False not supported yet')
        # return load_images(image_path, num_workers=num_workers)

    print('Loading Visual')
    visuals = load_visual(args)

    cache_exists = cache_path.is_file()
    region_cache_exists = cache_by_region_path.is_file()

    if cache_exists:
        print("Loading Image Cache")
        image_vectors = load_pickle(cache_path)

    if region_cache_exists:
        print("Loading Person Region Cache")
        region_vectors_by_episode = load_pickle(cache_by_region_path)
    
    if not cache_exists or not region_cache_exists:
        print("Loading Image Files and Building Image / Person Region Cache")

        if not cache_exists:
            image_vectors = {}

        if not region_cache_exists:
            region_vectors_by_episode = dict_for_each_episode()

        # Load visual data and model
        print('Loading extractor model')
        
        model = get_model(device)

        episode_paths = list(image_path.glob('*'))
        for e in tqdm(episode_paths, desc='Episode'):
            shot_paths = list(e.glob('*/*'))  # episode/scene/shot

            # Load image and flatten
            images = load_images(shot_paths, num_workers)
            images = {"{}{}{}".format(vid, delimiter, name): image for vid, shots in images.items() for name, image in shots.items()}

            dataset = ObjectDataset(args, images, transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))

            if not cache_exists:
                image_vectors_chunk = extract_features(
                    args, dataset, model, device=device, num_workers=num_workers, 
                    extractor_batch_size=args.extractor_batch_size
                )
                image_vectors = {**image_vectors, **image_vectors_chunk} # image_vectors.update(image_vectors_chunk)

            if not region_cache_exists:
                dataset.set_visual(visuals)
                region_chunk = extract_region_features(
                    args, dataset, model, device=device, num_workers=num_workers, 
                    extractor_batch_size=args.extractor_batch_size
                )
                for total, part in zip(region_vectors_by_episode, region_chunk):
                    total.update(part)

            del images
            del dataset

        del model

        image_vectors = merge_scene_features(image_vectors) 

        if cache:
            if not cache_exists:
                print("Saving Image Cache")
                save_pickle(image_vectors, cache_path)

            if not region_cache_exists:
                print("Saving Image Region Cache")
                save_pickle(region_vectors_by_episode, cache_by_region_path)

    if cache_by_episode_path.is_file():
        print("Loading Image Cache by Episode")
        image_vectors_by_episode = load_pickle(cache_by_episode_path)
    else:
        print('Merging Features by Episode')
        image_vectors_by_episode = merge_episode_features(image_vectors)

        if cache:
            print("Saving Image by Episode Cache")
            save_pickle(image_vectors_by_episode, cache_by_episode_path)

    return image_vectors, image_vectors_by_episode, region_vectors_by_episode, visuals

def load_images(shot_paths, num_workers=1):
    """
    images = {
        shot1: {
            frame_id1 : PIL image1, 
            frame_id2 : PIL image2, ...
        },

        shot2: {
            frame_id1 : PIL image1, 
            frame_id2 : PIL image2, ...
        },

        ...
    } # dictionary 
    """

    images = list(tqdm(map(load_image, shot_paths), total=len(shot_paths), desc='loading images'))
    images = {k: v for k, v in images}

    return images



def load_image(shot_path):
    """
    res = {
        frame_id1 : PIL image1, 
        frame_id2 : PIL image2, ...
    } # dictionary 

    return value = (
        shot, 
        {
            frame_id1 : PIL image1, 
            frame_id2 : PIL image2, ...
        }
    ) # tuple of shot and dictionary
    """

    image_paths = shot_path.glob('*')
    vid = '_'.join(shot_path.parts[-3:])
    # vid = shot_path.parts[-1]
    res = {}
    image_paths = sorted(list(image_paths))
    for image_path in image_paths:
        name = image_path.parts[-1] # name ex) IMAGE_0000046147.jpg
        image = Image.open(image_path)
        # image = image.resize(image_size) # moved to transform.Resize
        res[name] = image


    return (vid, res)


def load_visual(args):
    visual = load_json(args.visual_path)
    visual_by_episode = dict_for_each_episode()

    for shot, frames in visual.items():
        episode = get_episode_id(shot)
        episode_dict = visual_by_episode[episode]

        for frame in frames:
            frame_id = int(frame['frame_id'][-10:]) # 
            episode_dict[frame_id] = frame

    return visual_by_episode

class ObjectDataset(VisionDataset):
    def __init__(self, args, images, **kwargs):
        super(ObjectDataset, self).__init__('~/', **kwargs)

        self.args = args
        self.images = list([(k, v) for k, v in images.items()])
        self.visuals = None

    def set_visual(self, visuals):
        self.visuals = visuals

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        key, pil_img = self.images[idx]

        if self.transform is not None:
            tensor = self.transform(pil_img)  

        # Extract feature for each person region
        episode = get_episode_id(key)
        visual_key = get_frame_id(key)

        person_regions = []
        person_ids = []

        if self.visuals is not None:
            visuals = self.visuals[episode]
            if visual_key in visuals:
                persons = visuals[visual_key]["persons"]

                for p in persons:
                    person_id = p['person_id']
                    full_rect = p["person_info"]["full_rect"]

                    if full_rect["max_x"] != '':
                        top_left_v = full_rect["min_y"]
                        top_left_h = full_rect["min_x"]
                        height = full_rect["max_y"] - top_left_v
                        width = full_rect["max_x"] - top_left_h

                        region = transforms.functional.crop(pil_img, top_left_v, top_left_h, height, width)
                    else: # no bounding box
                        region = pil_img

                    if self.transform is not None:
                        region = self.transform(region)

                    person_regions.append(region)
                    person_ids.append(person_id)

            if not person_regions: # empty (no visual data or no person) - use all image
                person_regions.append(tensor)
                person_ids.append('None')

            person_regions = torch.stack(person_regions)

        return key, tensor, person_regions, person_ids

    def collate_fn(self, batch):
        key, tensor, person_regions, person_ids = zip(*batch)

        tensor = torch.stack(tensor)
        person_regions = list(person_regions)

        return key, tensor, person_regions, person_ids

def get_pooling(args):
    pooling = {
        'max': lambda x, dim: torch.max(x, dim=dim, keepdim=False)[0],
        'mean': lambda x, dim: torch.mean(x, dim=dim, keepdim=False),
    }[args.feature_pooling_method]

    return pooling

def extract_features(args, dataset, model, device=-1, num_workers=1, extractor_batch_size=384):
    """
    before: 
    images = {
        shot1 (vid) : {
            frame_id1 (name) : PIL image1 (image), 
            frame_id2 : PIL image2, ...
        } (shots),

        shot2: {
            frame_id1 : PIL image1, 
            frame_id2 : PIL image2, ...
        },

        ...
    } # dictionary 


    intermediate: 
    images = {
        shot/frame_id1 : PIL image1, (ex) AnotherMissOh04_042_1149/IMAGE_0000067819.jpg : <PIL.Image.Image ... >
        shot/frame_id2 : PIL image2,
        ...
    } # dictionary 


    after:

    images = {
        shot1 (vid) : {                       (ex) 'AnotherMissOh04_042_1149' :
            frame_id1 (name) : tensor1 (v),   (ex) 'IMAGE_0000067819.jpg': tensor(...)
            frame_id2 : tensor2,
            ...
        } (dt) ,

        shot2 : {                  
            frame_id1 : tensor1,   
            frame_id2 : tensor2,
            ...
        },

        ...
    }  
    """

    dataloader = utils.data.DataLoader(
        dataset,
        batch_size=extractor_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn
    )

    model.eval()
    pooling = get_pooling(args)
    images = {}

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=math.ceil(len(dataset) / extractor_batch_size), desc='extracting features'):
            keys, tensor, _, _ = data

            tensor = tensor.to(device)
            features = model(tensor).cpu()

            for key, feature, in zip(keys, features):
                vid, name = key.split(delimiter)
                images.setdefault(vid, {}).__setitem__(name, feature)
    
    del dataloader

    image_vectors = {}
    
    for vid, dt in images.items():
        feature = []
        frame_ids = []

        for name, v in dt.items():
            feature.append(v.to(device))
            frame_ids.append(name)

        feature = torch.stack(feature, dim=0)
        feature = pooling(feature, -1)
        feature = pooling(feature, -1)  # pooling image size
        # feature = pooling(feature, 0)  # pooling all images in shot (or scene)

        image_vectors[vid] = {'vectors': feature.cpu().numpy(), 'frame_ids': frame_ids}

    """
    image_vectors = {
        shot1 (vid) : {                       
            'vectors' : feature,     (shape: (# of frames in a shot, 512))
            'frame_ids' : frame_ids,  
        },

        shot2 : {                       
            'vectors' : feature,     
            'frame_ids' : frame_ids,  
        },
        ...
    }  
    """

    return image_vectors


def extract_region_features(args, dataset, model, device=-1, num_workers=1, extractor_batch_size=384):
    dataloader = utils.data.DataLoader(
        dataset,
        batch_size=extractor_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn
    )

    model.eval()
    pooling = get_pooling(args)
    regions = dict_for_each_episode()
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=math.ceil(len(dataset) / extractor_batch_size), desc='extracting features'):
            keys, _, person_regions, person_ids = data

            for j, region in enumerate(person_regions):
                region = region.to(device)
                region = model(region) # N X C x H x W (N: number of person regions in a frame)
                region = pooling(region, -1) # N x C x H
                region = pooling(region, -1) # N x C
                region = region.cpu().numpy()
                person_regions[j] = list(region)

            for key, region, person_id in zip(keys, person_regions, person_ids):
                vid, name = key.split(delimiter)

                e = get_episode_id(vid)
                f = get_frame_id(name)

                regions[e][f] = {
                    'person_id': person_id,
                    'image': region,
                }

    del dataloader

    return regions


def get_empty_image_vector(sample_image_size=[]):
    return torch.zeros(sample_image_size).cpu().numpy()


def get_model(device):
    # print("Using Resnet18")
    model = models.resnet18(pretrained=True)
    extractor = nn.Sequential(*list(model.children())[:-2])
    extractor.to(device)

    return extractor


def merge_scene_features(image_vectors):
    # mean pool
    keys = list(image_vectors.keys())
    scene = defaultdict(list)
    for key in keys:
        scene_key = '_'.join(key.split('_')[:-1] + ['0000'])
        scene[scene_key].append(torch.from_numpy(image_vectors[key]['vectors']))
    for k, v in scene.items():
        image_vectors[k] = torch.cat(v, dim=0).numpy()

    return image_vectors

def merge_episode_features(image_vectors):
    """
    features_by_episode = [
        {}, # empty dict

        { (episode1)
            frame_id1: vector1,
            frame_id2: vector2,
            ...
        },

        { (episode2)
            frame_id1: vector1, 
            frame_id2: vector2, 
            ...
        },

        ...

        { (episode18)
            frame_id1: vector1, 
            frame_id2: vector2, 
            ...
        }
    ]
    """

    features_by_episode = dict_for_each_episode()

    for shot, img in image_vectors.items():
        if shot.endswith('0000'):
            continue

        features = img['vectors']
        frame_ids = img['frame_ids']
        episode = get_episode_id(shot) # shot format: AnotherMissOh00_000_0000
        episode_dict = features_by_episode[episode]

        for i in range(len(features)):
            frame_id = get_frame_id(frame_ids[i])
            episode_dict[frame_id] = features[i]

    return features_by_episode


def dict_for_each_episode():
    return [dict() for i in range(18 + 1)]  # episode index: from 1 to 18