from functools import partial
from collections import defaultdict

import torch
from torchtext import data
import nltk

from utils import pad_tensor, make_jsonl, get_episode_id, get_scene_id
from .load_subtitle import merge_qa_subtitle, empty_sub
from .preprocess_image import preprocess_images, get_empty_image_vector

import json
import os
import re
from tqdm import tqdm
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from munch import Munch

import pickle
import copy

# debug 
from pprint import pprint

modes = ['train', 'val', 'test']

sos_token = '<sos>'
eos_token = '<eos>'
pad_token = '<pad>'
unk_token = '<unk>'

speaker_name = [
    'None', # index 0: unknown speaker 
    'Anna', 'Chairman', 'Deogi', 'Dokyung', 'Gitae',
    'Haeyoung1', 'Haeyoung2', 'Heeran', 'Hun', 'Jeongsuk',
    'Jinsang', 'Jiya', 'Kyungsu', 'Sangseok', 'Seohee', 
    'Soontack', 'Sukyung', 'Sungjin', 'Taejin', 'Yijoon'
]
speaker_index = {name: index for index, name in enumerate(speaker_name)} 
n_speakers = len(speaker_name)

# regex compiled in advance
nonword_re = re.compile('[\W]')
parentheses_re = re.compile(r'\(.*?\)') # (...) (minimal)
eos_re = re.compile(r'[\s]*[.?!]+[\s]*')

# torch datatype
int_dtype = torch.long
float_dtype = torch.float

def load_json(path):
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def get_data_path(args, mode='train', ext='.json'):
    name = args.data_path.name.split('_')
    name.insert(1, mode)
    name = '_'.join(name)
    path = args.data_path.parent / name
    path = path.parent / (path.stem + ext)

    return path

def get_vocab_attr_path(path, attr):
    return path.parent / (path.stem + '_' + attr + path.suffix)

def save_vocab(vocab, path):
    save_pickle(vocab, path)
    save_pickle(vocab.idx2word, get_vocab_attr_path(path, 'idx2word'))
    save_pickle(vocab.word2idx, get_vocab_attr_path(path, 'word2idx'))
    save_pickle(vocab.specials, get_vocab_attr_path(path, 'specials'))

def load_vocab(path):
    if not os.path.isfile(path):
        print('There is no cached vocab.')

        return None

    print('Using cached vocab.')

    vocab = load_pickle(path)
    idx2word, word2idx, specials = load_vocab_attr(path)

    set_vocab_attr(vocab, idx2word, word2idx, specials)

    return vocab

def load_vocab_attr(path):
    idx2word = load_pickle(get_vocab_attr_path(path, 'idx2word'))
    word2idx = load_pickle(get_vocab_attr_path(path, 'word2idx'))
    specials = load_pickle(get_vocab_attr_path(path, 'specials'))

    return idx2word, word2idx, specials


def get_frame_id(img_file_name):
    return int(img_file_name[-10:]) # img_file_name format: IMAGE_0000070227


class ImageData:
    def __init__(self, args):
        self.args = args
        self.image_dim = args.image_dim
        self.allow_empty_images = args.allow_empty_images
        self.num_workers = args.num_workers
        self.device = args.device
        self.image_dt, self.region_dt, self.visuals = self.load_images(
            args.image_path,
            cache=args.cache_image_vectors, 
            device=args.device
        )

        self.structure = self.build_structure()

        # will be set in self.attach_vocab and self.attach_visual
        # self.episode_dt = None 
        self.vocab = None 
        self.pad_index = None
        self.none_index = None
        self.visual_pad = None

    def load_images(self, image_path, cache=True, device=-1):
        # images, images_by_episode, = preprocess_images(self.args, image_path, cache=cache, device=device, num_workers=self.num_workers)
        images, images_by_episode, regions_by_episode, visuals = preprocess_images(self.args, image_path, cache=cache, device=device, num_workers=self.num_workers)

        if self.allow_empty_images:
            raise ValueError('Empty images are not allowed yet. ' \
                             'In config.py, change the value of allow_empty_images to False.')

        # print('Sorting images by frame id')
        # for scene_or_shot, frames in tqdm(images.items()):
        #     images[scene_or_shot] = [feature for shot, feature in sorted(frames.items())] 

            # Previous Implementation
            # for k, v in images.items():
            #     sample_image = v
            #     break
            # func = partial(get_empty_image_vector, sample_image_size=list(sample_image.shape))
            # images = defaultdict(func, images)

        # valid_frame_id = [[]] # leave valid_frame_id[0] as an empty list
        # valid_frame_id += [sorted(e.keys()) for e in images_by_episode[1:]]
        # print(valid_frame_id)

        # return images_by_episode
        return images_by_episode, regions_by_episode, visuals

    def build_structure(self):
        image_path = self.args.image_path
        episode_dirs = sorted(e for e in os.listdir(image_path) if e.startswith("AnotherMissOh"))

        episodes = {}

        for e in episode_dirs:
            episodes[get_episode_id(e)] = {}
            scenes = episodes[get_episode_id(e)]

            episode_path = image_path / e
            scene_dirs = sorted(s for s in os.listdir(episode_path) if s.isnumeric())

            for s in scene_dirs:
                scenes[int(s)] = {}
                shots = scenes[int(s)]

                scene_path = episode_path / s
                shot_dirs = sorted(sh for sh in os.listdir(scene_path) if sh.isnumeric())

                for sh in shot_dirs:
                    shot_path = scene_path / sh
                    shots[int(sh)] = sorted(get_frame_id(f.split('.')[0]) for f in os.listdir(shot_path) if f.startswith('IMAGE'))

        return episodes

    def attach_vocab(self, vocab):
        self.vocab = vocab
        self.pad_index = vocab.get_index(pad_token)
        self.none_index = speaker_index['None']

        self.attach_visual()

    def attach_visual(self):
        """
        After: 

        self.image_dt = [
            {}, # empty dict

            { (episode1)
                frame_id1: {
                    vector: vector1, 
                    persons: [
                        [person1_id_idx, behavior1_idx, emotion1_idx], # torch.Tensor
                        [person2_id_idx, behavior2_idx, emotion2_idx], # torch.Tensor
                        ...
                    ],
                },
                frame_id2: {
                    vector: vector2,
                    persons: [
                        [person1_id_idx, behavior1_idx, emotion1_idx], # torch.Tensor
                        [person2_id_idx, behavior2_idx, emotion2_idx], # torch.Tensor
                        ...
                    ],
                },
                ...
            },

            ...

            { (episode18)
                ...
            }
        ]
        """

        self.visual_pad = [self.none_index, self.pad_index, self.pad_index] 

        for frames in self.image_dt:
            for key, value in frames.items():
                frames[key] = {
                    'vector': value,
                    'persons': [],
                    'person_full': []
                }

        for e in range(1, 18 + 1):
            master_dict = self.image_dt[e]
            region_dict = self.region_dt[e]
            visual_dict = self.visuals[e]

            for frame, info in master_dict.items():
                if frame not in visual_dict:
                    continue

                visual = visual_dict[frame]
                processed_persons = master_dict[frame]['persons']
                for person in visual["persons"]:
                    person_id = person['person_id'].title()
                    person_id_idx = self.none_index if person_id == '' else speaker_index[person_id] # none -> None

                    person_info = person['person_info']

                    behavior = person_info['behavior'].lower()
                    behavior_idx = self.pad_index if behavior == '' else self.vocab.get_index(behavior.split()[0]) 

                    emotion = person_info['emotion'].lower()
                    emotion_idx= self.pad_index if emotion == '' else self.vocab.get_index(emotion) 

                    processed = [person_id_idx, behavior_idx, emotion_idx] # Don't convert visual to a tensor yet
                    processed_persons.append(processed)

                # when processed_persons is empty, region_dict[frame]['image'] contains 
                # features of all image. Just ignore this
                if processed_persons: # not empty
                    master_dict[frame]['person_full'] = region_dict[frame]['image']

                # assert len(master_dict[frame]['person_full']) == len(processed_persons), visual

    def get_image_by_vid(self, vid, shot_contained):
        episode = get_episode_id(vid)
        scene = get_scene_id(vid)
        first_shot = shot_contained[0]
        last_shot = shot_contained[-1]

        first_frame = self.structure[episode][scene][first_shot][0]
        last_frame = self.structure[episode][scene][last_shot][-1]

        return self.get_image_by_frame(episode, first_frame, last_frame + 1) # add + 1 to include last_frame


    def get_image_by_frame(self, episode, start_frame_id, end_frame_id):
        frames_in_episode = self.image_dt[episode]
        regions_in_episode = self.region_dt[episode]

        cnt = mean_feature = 0
        all_feature = []
        sample_vis = [] # becomes the first sample_vis data in the range
        all_vis = []
        all_p_full = []

        cur_id = start_frame_id
        added_8 = False

        while cur_id < end_frame_id:
            if cur_id in frames_in_episode: # found a frame in a certain shot
                frame = frames_in_episode[cur_id]

                p = frame['persons']
                sample_vis = p if sample_vis == [] else sample_vis
                all_vis.extend(p) # include all visual info of a frame

                f = frame['vector']
                mean_feature += f
                all_feature.extend(f for i in range(len(p)))

                pf = frame['person_full']
                all_p_full.extend(pf)

                cnt += 1

                # adjacent frame ids in a shot differ by 8, so
                # add 8 to cur_id to go to the next frame directly
                cur_id += 8 
                added_8 = True
            else:
                if added_8:
                    cur_id -= 8 # move back

                # increment by 1 until a frame in a certain shot is found
                cur_id += 1
                added_8 = False

        if cnt == 0: # start_id = end_id = -25 (no subtitle)
            mean_feature = np.zeros(self.image_dim)
        else:
            mean_feature /= cnt

        if not all_feature:
            all_feature.append(np.zeros(self.image_dim))

        if not sample_vis: # empty
            sample_vis = self.visual_pad
        else:
            sample_vis = sample_vis[0] # just select the first one

        if not all_vis: # empty
            all_vis = [self.visual_pad]

        if not all_p_full: # empty at the same time as all_vis
            all_p_full.append(np.zeros(self.image_dim))

        # assert len(all_vis) == len(all_p_full), (len(all_vis), len(all_p_full))

        # Don't convert visual to a tensor yet
        return mean_feature, all_feature, sample_vis, all_vis, all_p_full


class TextData(Dataset):
    def __init__(self, args, image, vocab=None, mode='train'):
        if mode not in modes:
            raise ValueError("mode should be %s." % (' or '.join(modes)))

        self.args = args
        self.mode = mode
        self.device = args.device

        # TODO: Check config file for input keys
        self.line_keys = ['que']  # 'description'
        self.list_keys = ['answers']
        self.contained_subs_keys = ['speaker', 'utter']

        self.max_sen_len = args.max_sentence_len
        self.max_sub_len = args.max_sub_len
        self.max_image_len = args.max_image_len

        self.glove_path = args.glove_path
        self.json_data_path = {m: get_data_path(args, mode=m, ext='.json') for m in modes}
        #self.pickle_data_path = {m: get_data_path(args, mode=m, ext='.pickle') for m in modes}
        self.pickle_data_path = {
            'train': get_data_path(args, mode='train', ext='.pickle'),
            'val': get_data_path(args, mode='val1', ext='.pickle'),
            'test': get_data_path(args, mode='test', ext='.pickle'),
        }
        self.vocab_path = args.vocab_path

        self.tokenizer = get_tokenizer(args)

        self.speaker_name = speaker_name
        self.speaker_index = speaker_index
        self.none_index = speaker_index['None']

        self.video_type_mapping = {'shot': 0, 'scene': 1}


        # print("using {} videoType".format(args.video_type))
        # for t in text_data:
        #     self.remove_questions(t, args.video_type)

        self.special_tokens = [sos_token, eos_token, pad_token, unk_token]

        if vocab is None: # build vocabulary
            self.vocab, data = self.build_word_vocabulary()
            self.data = data[mode]
        else:
            self.vocab = vocab
            self.data = load_pickle(self.pickle_data_path[mode])

        self.embedding_dim = self.vocab.shape[-1]
        self.pad_index = self.vocab.get_index(pad_token)
        self.eos_index = self.vocab.get_index(eos_token)


        ###### Image ######
        self.image = image
        self.image_dim = image.image_dim


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        vid = data['vid']
        episode = get_episode_id(vid)
        que = data['que']
        ans = data['answers'] 
        subtitle = data['subtitle']
        correct_idx = data['correct_idx'] if self.mode != 'test' else None
        q_level_logic = data['q_level_logic']
        # q_level_mem = data['q_level_mem']
        shot_contained = data['shot_contained'] 
        # video_type = self.video_type_mapping[data['videoType']] # 'shot': 0, 'scene': 1
        # qid = data['qid']

        # if len(shot_contained) == 1: # videoType == 'shot'
        #     shot_contained.append(shot_contained[0]) # ex) shot_contained = [26] -> [26, 26]

        speaker = []; sub = [] 
        image = []; all_image = []

        sample_visual = [] # just one visual
        sub_visual = [] # visual for each subtitle
        all_visual = [] # all visual in subtitles
        all_person_full = []

        if subtitle != empty_sub: # subtitle exists
            subs = subtitle["contained_subs"]

            for s in subs:
                st = s["st"]
                et = s["et"]
                if et - st > 90:
                    et = st + 3

                spkr = s["speaker"]
                utter = s["utter"]
                speaker.append(spkr)
                if len(utter) > self.max_sen_len:
                    del utter[self.max_sen_len:] # utter = utter[:self.max_sen_len]
                    utter[-1] = self.eos_index
                sub.append(utter)

                # img, all_img, sample_vis, all_vis, all_p_full = self.image.get_image_by_frame(episode, int(st * 25), int(et * 25))
                # image.append(img)
                # all_image.extend(all_img)
                # sample_visual.append(sample_vis)
                # sub_visual.append(all_vis)
                # all_visual.extend(all_vis)
                # all_person_full.extend(all_p_full)
            img, all_img, sample_vis, all_vis, all_p_full = self.image.get_image_by_vid(vid, shot_contained) # get all image in the scene/shot
            image.append(img)
            all_image.extend(all_img)
            sample_visual.append(sample_vis)
            sub_visual.append(all_vis)
            all_visual.extend(all_vis)
            all_person_full.extend(all_p_full)
        else: # No subtitle
            speaker.append(self.none_index) # add None speaker
            sub.append([self.pad_index]) # add <pad>
            img, all_img, sample_vis, all_vis, all_p_full = self.image.get_image_by_vid(vid, shot_contained) # get all image in the scene/shot
            image.append(img)
            all_image.extend(all_img)
            sample_visual.append(sample_vis)
            sub_visual.append(all_vis)
            all_visual.extend(all_vis)            
            all_person_full.extend(all_p_full)

        # Find who appears in que
        speaker_in_que = {idx for idx in que if idx < n_speakers}

        # If there is no speaker info in queue, or no masking, then use all subtitle 
        if not speaker_in_que or not self.args.mask: 
            speaker_in_que = set(speaker_index.values())

        # Filter data 
        filtered_sub = []; filtered_speaker = []
        filtered_vis = []; filtered_img = []; filtered_pf = []

        # Select subtitles whose speaker appears in que, then concatenate 
        n_words = 0
        for spkr, s in zip(speaker, sub):
            if spkr in speaker_in_que:
                sen_len = len(s)
                n_words += sen_len

                filtered_sub.extend(s)
                filtered_speaker.extend(spkr for i in range(sen_len)) # 1:1 correspondence between word and speaker

                if n_words > self.max_sub_len:
                    del filtered_sub[self.max_sub_len:] # filtered_sub = filtered_sub[:self.max_script_len]
                    filtered_sub[-1] = self.eos_index
                    del filtered_speaker[self.max_sub_len:] # filtered_speaker = filtered_speaker[:self.max_script_len]

                    break

        # Filter all_visual and all_image so that it only contains 
        # visual data and image of people who appear in que
        n_img = 0
        max_image_len = self.max_image_len
        for vis, img, pf in zip(all_visual, all_image, all_person_full):
            if vis[0] in speaker_in_que and vis != self.image.visual_pad:
                filtered_vis.extend(vis)
                filtered_img.append(img)
                filtered_pf.append(pf)
                n_img += 1
                if n_img > max_image_len:
                    del filtered_img[max_image_len:]
                    del filtered_vis[max_image_len * 3:]
                    del filtered_pf[max_image_len:]

                    break

        if not filtered_sub: # empty
            filtered_sub.append(self.pad_index)
            filtered_speaker.append(self.none_index)

        if not filtered_vis: # empty
            filtered_vis = self.image.visual_pad
            filtered_img = all_image

            if len(filtered_img) > max_image_len:
                del filtered_img[max_image_len:]

        if not filtered_pf:
            filtered_pf.append(np.zeros(self.image_dim))

            # if len(filtered_pf) > max_image_len:
            #     del filtered_pf[max_image_len:]

        # assert  3 * len(filtered_pf) == len(filtered_vis), (qid, len(filtered_pf), len(filtered_vis))

        data = {
            'vid': vid,
            # episode': episode,
            'que': que,
            'ans': ans,
            'sub': sub,
            'speaker': speaker,
            'correct_idx': correct_idx,
            'image': image,
            'all_image': all_image,
            'sample_visual': sample_visual,
            'sub_visual':sub_visual,
            'all_visual': all_visual,

            'filtered_speaker': filtered_speaker,
            'filtered_sub': filtered_sub,
            'filtered_image': filtered_img,
            'filtered_visual': filtered_vis,
            'filtered_person_full': filtered_pf,
            'q_level_logic': q_level_logic,
            # 'q_level_mem': q_level_mem,
            # 'shot_contained': shot_contained,
            # 'video_type': video_type
            # 'qid': qid
        }
        
        # currently not tensor yet
        return data

    def build_adjacency_matrix(self, batch_size, speaker, sub, sub_visual):
        adjacency = torch.zeros(batch_size, n_speakers, n_speakers, dtype=int_dtype) # 12 x 21 x 21

        for i in range(batch_size):
            for spkr, s, vs in zip(speaker[i], sub[i], sub_visual[i]):
                for idx in s:
                    # speaker GloVe index is equivalent to original speaker index
                    if idx < n_speakers: # person mentioned in subtitle
                        adjacency[i, spkr, idx] += 1

                        if spkr != idx: # prevent incremeting diagonal twice
                            adjacency[i, idx, spkr] += 1

                for v in vs:
                    visual_person = v[0] # 0th visual data refers to person_id
                    adjacency[i, spkr, visual_person] += 1

                    if spkr != visual_person: # prevent incremeting diagonal twice
                        adjacency[i, visual_person, spkr] += 1

        return adjacency

    # data padding
    def collate_fn(self, batch): 
        batch_size = len(batch)

        # episode = []; # vid = []; 
        # qid = []
        que = []; ans = []; correct_idx = []; 
        sub = []; speaker = []; # timestamp = []
        image = []; all_image = [];
        sample_visual = []; sub_visual = []; all_visual = []
        filtered_speaker = []; filtered_sub = []; filtered_image = []; filtered_visual = []; filtered_person_full = []
        q_level_logic = []; # q_level_mem = []; # video_type = []; # shot_contained = []

        for data in batch:
            # episode.append(data['episode'])
            que.append(data['que'])
            correct_idx.append(data['correct_idx'])
            ans.append(data['ans'])
            sub.append(data['sub'])
            speaker.append(data['speaker'])

            image.append(data['image'])
            all_image.append(data['all_image'])

            sample_visual.append(data['sample_visual'])
            sub_visual.append(data['sub_visual'])
            all_visual.append(data['all_visual'])

            filtered_speaker.append(data['filtered_speaker'])
            filtered_sub.append(data['filtered_sub'])
            filtered_image.append(data['filtered_image'])
            filtered_visual.append(data['filtered_visual'])
            filtered_person_full.append(data['filtered_person_full'])
            q_level_logic.append(data['q_level_logic'])
            # q_level_mem.append(data['q_level_mem'])
            # shot_contained.append(data['shot_contained'])
            # video_type.append(data['video_type'])
            # qid.append(data['qid'])

        # Build speaker adjacency matrix
        adjacency = self.build_adjacency_matrix(batch_size, speaker, sub, sub_visual)

        p_que, p_que_len = self.pad2d(que, self.pad_index, int_dtype)
        p_ans, _, p_ans_len = self.pad3d(ans, self.pad_index, int_dtype)
        p_correct_idx = torch.tensor(correct_idx, dtype=int_dtype) if self.mode != 'test' else None # correct_idx does not have to be padded
        
        p_speaker, _ = self.pad2d(speaker, self.none_index, int_dtype)
        p_image, _, _ = self.pad3d(image, 0, float_dtype)
        p_sample_visual, _, _ = self.pad3d(sample_visual, self.image.visual_pad, int_dtype)
        p_sub, p_sub_len, p_sub_sentence_len = self.pad3d(sub, self.pad_index, int_dtype)

        p_f_speaker, _ = self.pad2d(filtered_speaker, self.none_index, int_dtype)
        p_f_sub, p_f_sub_len = self.pad2d(filtered_sub, self.pad_index, int_dtype)
        p_f_visual, p_f_visual_len = self.pad2d(filtered_visual, self.image.visual_pad, int_dtype)
        p_f_image, p_f_image_len, _ = self.pad3d(filtered_image, 0, float_dtype)
        p_f_person_full, p_f_person_full_len, _ = self.pad3d(filtered_person_full, 0, float_dtype) 

        p_q_level_logic = q_level_logic # No converting to torch.Tensor
        # assert  (p_f_visual_len == 3 * p_f_person_full_len).all(), (p_f_visual_len, p_f_person_full_len)

        # p_episode = torch.tensor(episode, dtype=int_dtype) # episode does not have to be padded
        # p_q_level_logic = torch.tensor(q_level_logic, dtype=int_dtype) # q_level_logic does not have to be padded
        # p_q_level_mem = torch.tensor(q_level_mem, dtype=int_dtype) # q_level_mem does not have to be padded
        # p_shot_contained = torch.tensor(shot_contained, dtype=int_dtype) # shot_contained does not have to be padded
        # p_video_type = torch.tensor(video_type, dtype=int_dtype) # video_type does not have to be padded
        
        # Pack data into Munch to set additional attributes in ImageIterator
        data = Munch({
            'que': p_que,
            'answers': p_ans, 
            'subtitle': p_sub, 
            'speaker': p_speaker, 

            # 'episode': p_episode,
            'images': p_image,
            'sample_visual': p_sample_visual,
            # 'sub_visual': sub_visual,
            'filtered_visual': p_f_visual,
            'filtered_sub': p_f_sub,
            'filtered_speaker': p_f_speaker,
            'filtered_image': p_f_image,
            'filtered_person_full': p_f_person_full,
            'q_level_logic': p_q_level_logic,
            # 'q_level_mem': p_q_level_mem,
            # 'shot_contained': p_shot_contained,
            # 'video_type': p_video_type,
            # 'qid': qid,

            'que_len': p_que_len,
            'ans_len': p_ans_len,
            'sub_len': p_sub_len,
            'sub_sentence_len': p_sub_sentence_len,

            'filtered_visual_len': p_f_visual_len,
            'filtered_sub_len': p_f_sub_len,
            'filtered_image_len': p_f_image_len,
            'filtered_person_full_len': p_f_person_full_len,

            'adjacency': adjacency,
        })

        if p_correct_idx is not None:
            data['correct_idx'] = p_correct_idx

        return data

    def pad2d(self, data, pad_val, dtype):
        batch_size = len(data)
        length = [len(row) for row in data]
        max_length = max(length)
        shape = (batch_size, max_length)

        p_length = torch.tensor(length, dtype=int_dtype) # no need to pad

        if isinstance(pad_val, list):
            p_data = torch.tensor(pad_val, dtype=dtype)
            p_data = p_data.repeat(batch_size, max_length // len(pad_val))
        else:
            p_data = torch.full(shape, pad_val, dtype=dtype)

        for i in range(batch_size):
            d = torch.tensor(data[i], dtype=dtype)
            p_data[i, :len(d)] = d

        return p_data, p_length

    def pad3d(self, data, pad_val, dtype):
        batch_size = len(data)
        dim2_length = [[len(dim2) for dim2 in dim1] for dim1 in data]
        max_dim1_length = max(len(dim1) for dim1 in data)
        max_dim2_length = max(col for row in dim2_length for col in row)

        data_shape = (batch_size, max_dim1_length, max_dim2_length)
        
        p_dim2_length, p_dim1_length = self.pad2d(dim2_length, 0, int_dtype)

        if isinstance(pad_val, list):
            p_data = torch.tensor(pad_val, dtype=dtype)
            p_data = p_data.repeat(batch_size, max_dim1_length, max_dim2_length // len(pad_val))
        else:
            p_data = torch.full(data_shape, pad_val, dtype=dtype)

        for i in range(batch_size):
            row = data[i]
            for j in range(len(row)):
                d = torch.tensor(row[j], dtype=dtype)
                p_data[i, j, :len(d)] = d


        return p_data, p_dim1_length, p_dim2_length

    # borrowed this implementation from load_glove of tvqa_dataset.py (TVQA)
    def build_word_vocabulary(self, word_count_threshold=0):
        """borrowed this implementation from @karpathy's neuraltalk."""

        print("Building word vocabulary starts.\n")

        modes_str = "/".join(modes)

        # Load GloVe embedding
        print("Loading glove embedding at path: %s." % self.glove_path)

        glove_full, embedding_dim = self.load_glove(self.glove_path)
        glove_keys = glove_full.keys()

        print("Glove Loaded. Building vocabulary from %s data." % (modes_str))

        np.random.seed(0) 

        print('Merging QA and subtitles.')
        merge_text_data(self.args)

        texts = {mode: load_json(self.json_data_path[mode]) for mode in modes}

        all_sentences = []
        for text in texts.values():
            for e in text:
                for k in self.line_keys:
                    all_sentences.append(e[k])

                for k in self.list_keys:
                    all_sentences.extend(e[k])

                subtitle = e['subtitle']

                if subtitle is not empty_sub:
                    for sub in subtitle['contained_subs']:
                        for k in self.contained_subs_keys:
                            all_sentences.append(sub[k])

        visual = load_json(self.args.visual_path)
        text_in_visual = set()
        for frames in visual.values():
            for frame in frames:
                for person in frame["persons"]:
                    person_info = person['person_info']
                    text_in_visual.add(person_info['behavior'])
                    text_in_visual.add(person_info['emotion'])

        text_in_visual.remove('')
        all_sentences.extend(text_in_visual)

        # Find all unique words and count their occurence 
        word_counts = {}
        for sentence in all_sentences:
            for w in self.line_to_words(sentence, sos=False, eos=False, downcase=True):
                word_counts[w] = word_counts.get(w, 0) + 1

        n_all_words = len(word_counts)
        print("The number of all unique words in %s data: %d." % (modes_str, n_all_words))


        # Remove words that have no Glove embedding vector, or speaker names.
        # Speaker names will be added later with random vectors. 
        unk_words = [w for w in word_counts if w not in glove_keys or w.title() in self.speaker_name]
        for w in unk_words:
            del word_counts[w]

        n_glove_words = len(word_counts)
        n_unk_words = n_all_words - n_glove_words
        print("The number of all unique words in %s data that uses GloVe embeddings: %d. "
              '%.2f%% words are treated as %s or speaker names' 
              % (modes_str, n_glove_words, 100 * n_unk_words / n_all_words, unk_token))

        # Accept words whose occurence counts are greater or equal to the threshold.
        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold and w not in self.special_tokens]
        print("Vocabulary Size %d (%s excluded) using word_count_threshold %d.\n" %
              (len(vocab), ' '.join(self.special_tokens), word_count_threshold))

        # Build index and vocabularies.
        print("Building word2idx, idx2word mapping.")

        # speaker name
        word2idx = {name.lower(): idx for name, idx in self.speaker_index.items()}
        idx2word = {idx: token for token, idx in word2idx.items()}

        offset = len(word2idx)

        # special characters
        for idx, w in enumerate(self.special_tokens):
            word2idx[w] = idx + offset
            idx2word[idx + offset] = w

        offset = offset + len(self.special_tokens)

        # # special characters
        # word2idx = {token: idx for idx, token in enumerate(self.special_tokens)}
        # idx2word = {idx: token for token, idx in word2idx.items()}
        # offset = len(self.special_tokens)

        # # speaker name
        # for idx, w in enumerate(self.speaker_name):
        #     w = w.lower()
        #     word2idx[w] = idx + offset
        #     idx2word[idx + offset] = w

        # offset = offset + len(self.speaker_name)

        for idx, w in enumerate(vocab):
            word2idx[w] = idx + offset
            idx2word[idx + offset] = w

        print("word2idx size: %d, idx2word size: %d.\n" % (len(word2idx), len(idx2word)))

        # Build GloVe matrix
        glove_matrix = np.zeros([len(idx2word), embedding_dim])
        n_glove = n_unk = n_name = n_zero = 0
        unk_words = []
        for i in tqdm(range(len(idx2word))):
            w = idx2word[i]

            if w.title() in self.speaker_name[1:]: # Remove 'None' from speaker name to use GloVe vector.
                w_embed = np.random.randn(embedding_dim) * 0.4
                n_name += 1
            elif w in glove_keys:
                w_embed = glove_full[w]  
                n_glove += 1
            elif w == pad_token: 
                w_embed = 0
                n_zero += 1
            else: # <eos>, <sos> are all mapped to <unk>
                w_embed = glove_full[unk_token]
                n_unk += 1 
                unk_words.append(w)

            glove_matrix[i, :] = w_embed


        print("Vocab embedding size is :", glove_matrix.shape)
        print('%d words are initialized with known GloVe vectors, '\
              '%d words (names) are randomly initialized, '\
              '%d words (%s) are initialized as 0, and '\
              '%d words (%s) are initialized with %s GloVe vectors.' \
              % (n_glove, n_name, n_zero, pad_token, n_unk, ' '.join(unk_words), unk_token))

        print("Building vocabulary done.")

        vocab = torch.Tensor(glove_matrix) 

        # Add attributes to vocab (for extended functionality and compatibility)
        set_vocab_attr(vocab, idx2word, word2idx, self.special_tokens)

        print("Saving vocab as pickle.")
        save_vocab(vocab, self.vocab_path)

        print('Splitting long subtitles and converting words in text data to indices, timestamps from string to float.')
        for mode, text in texts.items():
            texts[mode] = self.preprocess_text(text, vocab)

        print("Saving converted data as pickle.")
        for mode in modes:
            save_pickle(texts[mode], self.pickle_data_path[mode]) 

        return vocab, texts

    # borrowed this implementation from load_glove of tvqa_dataset.py (TVQA)
    @classmethod
    def load_glove(cls, glove_path):
        glove = {}

        with open(glove_path, encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                values = line.strip('\n').split(' ')
                word = values[0]
                vector = np.asarray([float(e) for e in values[1:]])
                glove[word] = vector

        embedding_dim = len(vector)

        return glove, embedding_dim

    def split_subtitle(self, sub, sos=True, eos=True, to_indices=False, word2idx=None):
        if to_indices == True and word2idx == None:
            raise ValueError('word2idx should be given when to_indices is True')

        n_special_tokens = sos + eos # True == 1, False == 0

        st, et = sub['st'], sub['et']
        t_range = et - st

        speaker = sub['speaker']

        utters = self.split_string(sub['utter'], sos=sos, eos=eos)
        if to_indices:
            utters = [self.words_to_indices(words, word2idx) for words in utters] # 

        if len(utters) == 1: 
            sub['utter'] = utters[0]

            return [sub] 

        utters_len = np.array([len(u) - n_special_tokens for u in utters]) # -2 for <sos> and <eos>
        ratio = utters_len.cumsum() / utters_len.sum()
        ets = st + ratio * t_range
        sts = [st] + list(ets[:-1])

        subs = [dict(speaker=speaker, st=s, et=e, utter=u) for s, e, u in zip(sts, ets, utters)]

        return subs

    
    # Split a string with multiple sentences to strings with one sentence.
    def split_string(self, string, min_sen_len=3, sos=True, eos=True):
        split = eos_re.split(string)
        split = list(filter(None, split)) # remove '' 
        split = [self.line_to_words(s, sos=sos, eos=eos) for s in split] # tokenize each split sentence

        # Merge short sentences to adjacent sentences
        n_special_tokens = sos + eos # True == 1, False == 0

        no_short = []
        i = 0
        n_sentences = len(split)
        while i < n_sentences:
            length = len(split[i]) - n_special_tokens # -2 for <sos> and <eos>
            if length < min_sen_len: 
                if i == 0:
                    if n_sentences == 1:
                        s = split[i] # 0
                    else:
                        # concatenate split[0] and split[1]
                        # if eos == True (== 1), exclude <eos> from split[0] (split[i][:-1])
                        # else                 ,           just use split[0] (split[i][:len(split[i])])
                        # 
                        # if sos == True (== 1), exclude <sos> from split[1] (split[i + 1][1:]) 
                        # else                 ,           just use split[1] (split[i + 1][0:]) 
                        s = split[i][:len(split[i])-eos] + split[i + 1][sos:] 
                        i += 1

                    no_short.append(s)
                else:
                    no_short[-1] = no_short[-1][:len(no_short[-1])-eos] + split[i][sos:] 
            else:
                s = split[i]
                no_short.append(s)
                
            i += 1

        return no_short


    # borrowed this implementation from line_to_words of tvqa_dataset.py (TVQA)
    def line_to_words(self, line, sos=True, eos=True, downcase=True):
        line = clean_string(line)

        tokens = self.tokenizer(line.lower()) if downcase else self.tokenizer(line)

        words = [sos_token] if sos else []
        words = words + [w for w in tokens if w != ""]
        words = words + [eos_token] if eos else words

        return words

    def words_to_indices(self, words, word2idx):
        indices = [word2idx.get(w, word2idx[unk_token]) for w in words]

        return indices

    def line_to_indices(self, line, word2idx, sos=True, eos=True, downcase=True):
        words = self.line_to_words(line, sos=sos, eos=eos, downcase=downcase)
        indices = self.words_to_indices(words, word2idx)

        return indices

    # TODO
    def remove_questions(self, dataset, video_types):
        pass
        # li = []
        # for example in dataset.examples:
        #     if example.videoType in video_types:
        #         li.append(example)

        # dataset.examples = li

    def preprocess_text(self, text, vocab):
        word2idx = vocab.word2idx

        for e in text:
            for k in self.line_keys:
                e[k] = self.line_to_indices(e[k], word2idx)

            for k in self.list_keys:
                e[k] = [self.line_to_indices(line, word2idx) for line in e[k]]

            subtitle = e['subtitle']

            if subtitle is not empty_sub:
                subtitle['et'] = float(subtitle['et'])
                subtitle['st'] = float(subtitle['st'])

                new_subs = []

                for sub in subtitle['contained_subs']:
                    sub['et'] = float(sub['et'])
                    sub['st'] = float(sub['st'])
                    sub['speaker'] = self.speaker_index[sub['speaker']] # to speaker index
                    split_subs = self.split_subtitle(sub, to_indices=True, word2idx=word2idx)
                    new_subs.extend(split_subs)

                subtitle['contained_subs'] = new_subs

        return text

def merge_text_data(args):
    for mode in modes:
        ext = '.json'
        new_path = get_data_path(args, mode=mode, ext=ext)
        qa_path = new_path.parent / (new_path.stem[:new_path.stem.find('_script')] + ext)
        subtitle_path = args.subtitle_path

        merge_qa_subtitle(new_path, qa_path, subtitle_path)
        make_jsonl(new_path)


# Add attributes to vocab (for extended functionality and compatibility)
def set_vocab_attr(vocab, idx2word, word2idx, special_tokens):
    setattr(vocab, 'idx2word', idx2word) # vocab.idx2word = idx2word
    setattr(vocab, 'itos', idx2word) # vocab.itos = idx2word (for compatibility)
    setattr(vocab, 'get_word', lambda idx: getattr(vocab, 'idx2word')[idx]) # vocab.get_word(idx) = vocab.idx2word[idx]

    setattr(vocab, 'word2idx', word2idx) # vocab.word2idx = word2idx
    setattr(vocab, 'stoi', word2idx) # vocab.stoi = word2idx (for compatibility)
    setattr(vocab, 'get_index', lambda word: getattr(vocab, 'word2idx').get(word, getattr(vocab, 'word2idx')[unk_token])) # vocab.get_index(word) = vocab.get_index.get(word, vocab.word2idx[unk_token])

    setattr(vocab, 'specials', special_tokens) # vocab.specials = self.special_tokens (['<sos>', '<eos>', '<pad>', '<unk>'])
    setattr(vocab, 'special_ids', [vocab.stoi[k] for k in vocab.specials]) # vocab.special_ids = 
    for token in vocab.specials:
        setattr(vocab, token[1:-1], token) # vocab.sos = '<sos>' ... 

    return vocab


def load_text_data(args, vocab=None):
    print('Load Image Data')
    image = ImageData(args)

    print('Load Text Data')
    if vocab is None:
        # Use cached vocab if it exists.
        vocab = load_vocab(args.vocab_path)

    train_dataset = TextData(args, image, vocab=vocab, mode='train')

    vocab = train_dataset.vocab
    image.attach_vocab(vocab)

    valid_dataset = TextData(args, image, vocab=vocab, mode='val')
    test_dataset  = TextData(args, image, vocab=vocab, mode='test')

    train_iter = DataLoader(
        train_dataset, 
        batch_size=args.batch_sizes[0],
        shuffle=args.shuffle[0], 
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn
    )

    val_iter = DataLoader(
        valid_dataset, 
        batch_size=args.batch_sizes[1],
        shuffle=args.shuffle[1],
        num_workers=args.num_workers,
        collate_fn=valid_dataset.collate_fn
    )

    test_iter = DataLoader(
        test_dataset, 
        batch_size=args.batch_sizes[2],
        shuffle=args.shuffle[2],
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn
    )

    # # test
    # print('-------- TESTING DATA LOADING --------')

    # # train_iter_test = next((iter(train_iter)))
    # # # list(print(key, value.shape) for key, value in train_iter_test.items() if isinstance(value, torch.Tensor))
    # # pprint(train_iter_test)
    # # print()

    # # import sys 
    # # print('Debug - Terminate')
    # # sys.exit()

    # # for key, value in train_iter_test.items():
    # #     print(key, value.shape)

    # print('Test Loading Train Text Data')
    # for data in tqdm(train_iter):
    #     pass


    # print('Test Loading Val Text Data')
    # for data in tqdm(val_iter):
    #     pass

    # print('Test Loading Test Text Data')
    # for data in tqdm(test_iter):
    #     pass

    # print('-------- TESTING DATA LOADING END --------')
    # print()

    return {'train': train_iter, 'val': val_iter, 'test': test_iter}, vocab

def nonword_tokenizer(sentence):
    return nonword_re.split(sentence)


def clean_string(string):
    # string = parentheses_re.sub('', string) # delete (...)

    string = re.sub(r"[^A-Za-z0-9!?.]", " ", string) # remove all special characters except ! ? .
    string = re.sub(r"(\.){2,}", ".", string) # re.sub(r"(\.)+", " . ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip()


def get_tokenizer(args):
    tokenizers = {
        'nltk': nltk.word_tokenize,
        'nonword': nonword_tokenizer,
    }

    return tokenizers[args.tokenizer.lower()]

# batch: [len, batch_size]
def get_iterator(args, vocab=None):
    iters, vocab = load_text_data(args, vocab)
    print("Data Loading Done")

    # return image_iters, vocab
    return iters, vocab
