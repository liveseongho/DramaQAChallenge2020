import os
import json
from pathlib import Path
from unidecode import unidecode

from tqdm import tqdm

# debug
from pprint import pprint

empty_sub = '.'

def load_subtitle(subtitle_path):
    subtitle_path = Path(subtitle_path)
    '''
    paths = list(subtitle_path.glob('*.json'))
    if len(paths) == 0:
        paths = list(subtitle_path.glob('*/*.json'))
    '''
    subtitles = {}
    speakers = {}
    with open(str(subtitle_path), 'r') as f:
        subtitles = json.load(f)
        for vid, v in subtitles.items():
            for sub in subtitles[vid]["contained_subs"]:
                sub["utter"] = unidecode(sub["utter"])
        #subtitles = {vid: ' '.join([subtitle['utter'] for subtitle in v['contained_subs']])
        #             for vid, v in subtitles.items()}
    return subtitles


def merge_qa_subtitle(new_path, qa_path, subtitle_path):
    #if not os.path.isfile(str(new_path)):
    if True:
        print("Processing subtitle data")

        # subtitles = subtitles.json
        subtitles = load_subtitle(subtitle_path)
        
        # qa = QA.json
        with open(str(qa_path), 'r') as f:
            qa = json.load(f)

        res = []
        for row in tqdm(qa):
            if row['vid'].endswith('_0000'):
                # scene question
                vid = row['vid']
                vid_prefix = vid[:vid.find('_0000')]

                shot_subtitles = []
                shot_st = []
                shot_et = []
                subtitle_sts = set()

                # if vid starts with vid_prefix,
                # add sub to shot_subtitles if the same sub has not been added yet
                for vid, subs in subtitles.items():
                    if not vid.startswith(vid_prefix):
                        continue 

                    shot_st.append(subs['st'])
                    shot_et.append(subs['et'])
                    
                    for sub in subs['contained_subs']:
                        st = sub['st']

                        if st in subtitle_sts:
                            continue

                        subtitle_sts.add(st)
                        shot_subtitles.append((float(st), sub))

                shot_st = sorted(shot_st, key=float)
                shot_et = sorted(shot_et, key=float)
                shot_subtitles.sort()

                # pprint(shot_subtitles)


                # shot_subtitles = sorted([(vid, sub) for vid, sub in subtitles.items()
                #             if vid.startswith(vid_prefix)])


                if shot_subtitles:
                    # row['subtitle'] = [sub["contained_subs"] for vid, sub in shot_subtitles ]

                    row['subtitle'] = {}
                    row['subtitle']["contained_subs"] = [sub for st, sub in shot_subtitles]
                    row['subtitle']["st"] = shot_st[0]
                    row['subtitle']["et"] = shot_et[-1]
                else:
                    row['subtitle'] = ''
                '''a
                speaker = sorted([(vid, sub) for vid, sub in speakers.items()
                            if vid.startswith(vid_prefix)])
                speaker = ','.join([v[1] for v in speaker])
                row['speaker'] = speaker
                '''
            else:
                # shot question
                if row['vid'] in subtitles:
                    row['subtitle'] = subtitles[row['vid']]
                else:
                    row['subtitle'] = ''
                '''
                if row['vid'] in speakers:
                    row['speaker'] = speakers[row['vid']]
                else:
                    row['speaker'] = ''
                '''
            if row['subtitle'] == '':
                row['subtitle'] = empty_sub  # prevent empty string
            res.append(row)

        

        with open(str(new_path), 'w') as f:
            json.dump(res, f, indent=4)
