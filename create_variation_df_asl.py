import pandas as pd
import json
import numpy as np

def create_var_df(asl_df):
    with open('archive/WLASL_v0.3.json', 'r') as file:
            ASL_dict = json.load(file)
    asl_glosses = []
    asl_videos = []
    asl_poses = []
    asl_hands = []
    asl_variation = []

    unique_glosses = sorted(asl_df['ASL_GLOSSES'].unique())
    for i in range(len(ASL_dict)):
        asl_gloss = ASL_dict[i]['gloss']
        if asl_gloss in unique_glosses:
            asl_instances = ASL_dict[i]['instances']
            current_row = asl_df[asl_df['ASL_GLOSSES'] == asl_gloss]
            video_list = list(current_row['ASL_VIDS'])[0]
            for j in range(len(asl_instances)):
                    video_id = asl_instances[j]['video_id']
                    if video_id in video_list:
                        index = video_list.index(video_id)
                        variation_id = asl_instances[j]['variation_id']
                        asl_glosses.append(f'{asl_gloss}_{variation_id}')
                        asl_videos.append(video_id)
                        poses = list(current_row['ASL_POSES'])[0][index]
                        hands = list(current_row['ASL_HANDS'])[0][index]
                        asl_poses.append(poses)
                        asl_hands.append(hands)
                        asl_variation.append(variation_id)
                        #   print(video_id, variation_id)

    var_asl_df = pd.DataFrame(data = {"ASL_GLOSSES":asl_glosses, "ASL_VIDS":asl_videos, "ASL_POSES":asl_poses, "ASL_HANDS":asl_hands, "ASL_VARIATION":asl_variation})
    return var_asl_df
                    