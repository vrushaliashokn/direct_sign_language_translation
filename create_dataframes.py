import pandas as pd
import json
import numpy as np
from datasets import load_dataset
import extractor
print("IN CREATE DATAFRAMES --------------------")

asl_train_list = np.load('ASL_WORDS.npy')
isl_train_list = np.load('ISL_WORDS.npy')
#-----------------------------------
# Create ASL dataframe
def create_ASL_dataframe(split = "train"):
    with open('wlasl-complete/WLASL_v0.3.json', 'r') as file:
        ASL_dict = json.load(file)

    asl_glosses = []
    asl_videos = []
    asl_poses = []
    asl_hands = []
    print("Total: ", len(ASL_dict))
    count = 0
    for i in range(len(ASL_dict)):
        print(f"No.: {i} ---------------------------------------------" )
        asl_gloss = ASL_dict[i]['gloss']
        asl_video_ids = []
        asl_poses_list = []
        asl_hands_list = []
        asl_instances = ASL_dict[i]['instances']
        # final_len = min(4, len(asl_instances))
        for j in range(len(asl_instances)):
            pose_j = extractor.pose_extractor(asl_instances[j]['video_id'], "asl")
            hand_j = extractor.hand_extractor(asl_instances[j]['video_id'], "asl")
            if pose_j is not None and hand_j is not None:
                asl_video_ids.append(asl_instances[j]['video_id'])
                asl_poses_list.append(extractor.pose_extractor(asl_instances[j]['video_id'], "asl"))
                asl_hands_list.append(extractor.hand_extractor(asl_instances[j]['video_id'], "asl"))
        if len(asl_video_ids) != 0:
            count += 1
            asl_glosses.append(asl_gloss)
            asl_videos.append(asl_video_ids)
            asl_poses.append(asl_poses_list)
            asl_hands.append(asl_hands_list)
        # if count > 20:
        #     break

    asl_df = pd.DataFrame(data = {"ASL_GLOSSES":asl_glosses, "ASL_VIDS":asl_videos, "ASL_POSES":asl_poses, "ASL_HANDS":asl_hands})
    return asl_df

def create_DAI_dataframe(split = "train"):
    gloss_df = pd.read_csv("asllrp_sentence_signs_2025_06_28.csv", on_bad_lines='skip')
    glosses = np.unique(gloss_df['Main entry gloss label'])
    asl_glosses = []
    asl_videos = []
    asl_poses = []
    asl_hands = []
    for i, row in gloss_df.iterrows():
        gloss = row['Main entry gloss label'].lower()
        if gloss in isl_train_list:
            print(gloss)
            print(f"No.: {i} ---------------------------------------------" )
            video_name = row['Sign video filename'] 
            video_path = f"batch_signs_video_v3_1/{video_name}"
            pose = extractor.pose_extractor(video_path, "isl")
            hand = extractor.hand_extractor(video_path, "isl")
            asl_glosses.append(gloss)
            asl_videos.append(video_name)
            asl_poses.append(pose)
            asl_hands.append(hand)
    asl_df = pd.DataFrame(data = {"ASL_GLOSSES":asl_glosses, "ASL_VIDS":asl_videos, "ASL_POSES":asl_poses, "ASL_HANDS":asl_hands})
    return asl_df



#-----------------------------------
#Create ISL dataframe
def create_ISL_dataframe(split = "train"):
    ISL_dict = load_dataset("ai4bharat/INCLUDE")
    count = 0
    isl_glosses = []
    isl_videos = []
    isl_poses = []
    isl_hands = []
    print("Total: ", len(ISL_dict[split]))
    for i in range(len(ISL_dict[split])):
        isl_gloss_num = ISL_dict[split][i]['label']
        isl_gloss = isl_gloss_num[(isl_gloss_num.find(" ")+1):].lower()
        if isl_gloss in isl_train_list:
            isl_video_id = ISL_dict[split][i]['video_path']
            isl_poses.append(extractor.pose_extractor(ISL_dict[split][i]['video_path'], "isl"))
            isl_hands.append(extractor.hand_extractor(ISL_dict[split][i]['video_path'], "isl"))
            isl_glosses.append(isl_gloss)
            isl_videos.append(isl_video_id)
            count += 1

    isl_df = pd.DataFrame(data = {"ISL_GLOSSES":isl_glosses, "ISL_VIDS":isl_videos, "ISL_POSES":isl_poses, "ISL_HANDS":isl_hands})
    return isl_df

print("dai_asl_df")
dai_asl_df = create_DAI_dataframe(split = "train")
dai_asl_df.to_pickle("dai_asl.pkl")

# print("new")
# print("train_asl_df")
# train_asl_df = create_ASL_dataframe(split = "train")
# train_asl_df.to_pickle("wl-asl_complete_try.pkl")

# print("test_asl_df")
# test_asl_df = create_ASL_dataframe(split = "test")
# test_asl_df.to_pickle("ASL_dataset_test.pkl")

# print("train_isl_df")
# train_isl_df = create_ISL_dataframe(split = "train")
# train_isl_df.to_pickle("ISL_dataset_train_FULL.pkl")

# print("test_isl_df")
# test_isl_df = create_ISL_dataframe(split = "test")
# test_isl_df.to_pickle("ISL_dataset_test_FULL.pkl")

# print("new")
# print("complete_asl_df")
# complete_asl_df = create_ASL_dataframe(split = "train")
# complete_asl_df.to_pickle("Complete_ASL_DF.pkl")
