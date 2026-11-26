import pandas as pd
import numpy as np
import data_augment as augment
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import create_variation_df_asl as cvda
from sklearn.manifold import TSNE
from dataset import ASLDataset, ASLTestDataset, ISLDataset
from torch.utils.data import DataLoader
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.utils.rnn import pad_sequence

train_asl_df = pd.read_pickle("dai_asl.pkl")
class_counts = train_asl_df['ASL_GLOSSES'].value_counts()
valid_classes = class_counts[class_counts >= 3].index
train_asl_df = train_asl_df[train_asl_df['ASL_GLOSSES'].isin(valid_classes)]
print(len(np.unique(train_asl_df['ASL_GLOSSES'])))
print(np.unique(train_asl_df['ASL_GLOSSES'], return_counts = True))
# full_train_asl_df = cvda.create_var_df(train_asl_df)
# full_train_isl_df = pd.read_pickle("ISL_dataset_train_FULL.pkl")



# # counts = full_train_asl_df['ASL_GLOSSES'].value_counts().sort_values(ascending=True)
# counts = full_train_isl_df['ISL_GLOSSES'].value_counts().sort_values(ascending=True)

# print(f"Total classes:             {len(counts)}")
# print(f"Total samples:             {counts.sum()}")
# print(f"Mean per class:            {counts.mean():.1f}")
# print(f"Median per class:          {counts.median():.1f}")
# print(f"Min: {counts.min()} ({counts.idxmin()})")
# print(f"Max: {counts.max()} ({counts.idxmax()})")
# print(f"Classes with < 20 samples: {(counts < 20).sum()}")
# print(f"Classes with < 50 samples: {(counts < 50).sum()}")







# # # class_counts = full_train_asl_df['ASL_GLOSSES'].value_counts()
# # # valid_classes = class_counts[class_counts >= 8].index
# # # # valid_classes = np.unique(full_test_asl_df['ASL_GLOSSES'])
# # # print(len(full_train_asl_df))
# # # full_train_asl_df = full_train_asl_df[full_train_asl_df['ASL_GLOSSES'].isin(valid_classes)]
# # # print(len(full_train_asl_df))
# # # top10_classes = full_train_asl_df['ASL_GLOSSES'].value_counts().head(50).index
# # # print("Top 10 classes:", top10_classes.tolist())
# # # full_train_asl_df = full_train_asl_df[full_train_asl_df['ASL_GLOSSES'].isin(top10_classes)]
# # # full_test_asl_df  = full_test_asl_df[full_test_asl_df['ASL_GLOSSES'].isin(top10_classes)]

# # # print("=== RAW DATAFRAME ===")
# # # print("Total rows:", len(full_train_asl_df))
# # # print("\nValue counts:")
# # # print(full_train_asl_df['ASL_GLOSSES'].value_counts())

# # # print("\n=== TOP 10 CLASSES ===")
# # # print(top10_classes)
# # # print("Type:", type(top10_classes))

# # # print("\n=== AFTER FILTERING ===")
# # # filtered = full_train_asl_df[full_train_asl_df['ASL_GLOSSES'].isin(top10_classes)]
# # # print("Filtered rows:", len(filtered))
# # # print(filtered['ASL_GLOSSES'].value_counts())



# # # def normalize_hands(seq):
# # #     seq = np.nan_to_num(seq, nan=0.0)
# # #     seq = seq[:, :, :3]          # drop confidence channel
    
# # #     left  = seq[:, :21, :]       # (T, 21, 3)
# # #     right = seq[:, 21:, :]       # (T, 21, 3)
    
# # #     def normalize_one_hand(hand):
# # #         wrist = hand[:, 0:1, :]  # (T, 1, 3)
# # #         scale = np.linalg.norm(hand[:, 9, :] - hand[:, 0, :],
# # #                                axis=-1, keepdims=True)
# # #         scale = np.where(scale == 0, 1, scale)
# # #         return (hand - wrist) / scale[:, np.newaxis, :]
    
# # #     left_norm  = normalize_one_hand(left)
# # #     right_norm = normalize_one_hand(right)
    
# # #     return np.concatenate([left_norm, right_norm], axis=1)  # (T, 42, 3)

# # # def flatten_samples(df, feature_col='ASL_HANDS'):
# # #     X = []
# # #     for seq in df[feature_col]:
# # #         normalized = normalize_hands(seq)  # (T, 42, 3)
# # #         mean = normalized.mean(axis=0).flatten()   # (126,)
# # #         std  = normalized.std(axis=0).flatten()    # (126,) — captures motion
# # #         X.append(np.concatenate([mean, std]))      # (252,)
# # #     y = LabelEncoder().fit_transform(df['ASL_GLOSSES'])
# # #     return np.array(X), y

# # # X, y = flatten_samples(full_train_asl_df)
# # # print("X shape:", X.shape)  # should be (N, 126)

# # # X, y = flatten_samples(full_train_asl_df)

# # # # PCA first (fast)
# # # pca = PCA(n_components=2)
# # # X_2d = pca.fit_transform(X)
# # # print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# # # plt.figure(figsize=(10, 7))
# # # for i, gloss in enumerate(top10_classes):
# # #     mask = y == i
# # #     plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=gloss, alpha=0.6, s=30)
# # # plt.legend(bbox_to_anchor=(1.05, 1))
# # # plt.title('PCA of hand features per class')
# # # plt.tight_layout()
# # # plt.show()

# # # # test_counts = full_test_asl_df['ASL_GLOSSES'].value_counts()
# # # # print(len(full_test_asl_df))
# # # # full_test_asl_df = full_test_asl_df[full_test_asl_df['ASL_GLOSSES'].isin(valid_classes)]
# # # # print(len(full_test_asl_df))


# # # train_asl_df = pd.read_pickle("ASL_dataset_train_FULL.pkl")
# # # # print(train_asl_df.head())
# # # full_train_asl_df = cvda.create_var_df(train_asl_df)
# # # counts = full_train_asl_df['ASL_GLOSSES'].value_counts()

# # # plt.figure(figsize=(8, 5))
# # # plt.hist(counts, bins=20)

# # # plt.xlabel("Number of samples per class")
# # # plt.ylabel("Number of classes")
# # # plt.title("Histogram of Class Distribution")
# # # plt.show()
# # # counts = train_asl_df['ASL_GLOSSES'].value_counts(-50)

# # # plt.figure(figsize=(12, 6))
# # # plt.bar(counts.index, counts.values)

# # # plt.xticks(rotation=90)
# # # plt.xlabel("ASL Gloss")
# # # plt.ylabel("Count")
# # # plt.title("Class vs Count Distribution")

# # # plt.tight_layout()
# # # plt.show(
# # # counts = full_train_asl_df['ASL_GLOSSES'].value_counts()
# # # print(counts)
# # # # sorted_counts = counts.values  # already sorted descending

# # # # plt.figure(figsize=(10, 6))
# # # # plt.plot(sorted_counts)

# # # plt.xlabel("Class Rank (most frequent to least frequent)")
# # # plt.ylabel("Number of Samples")
# # # plt.title("Long-Tail Distribution of ASL Glosses")

# # # plt.grid(alpha=0.3)
# # # plt.show()

# # # top_classes = full_train_asl_df['ASL_GLOSSES'].value_counts().head(20)
# # top_classes = counts.head(20).index.tolist()


# # unique_glosses = sorted(few_classes_train['ASL_GLOSSES'].unique())
# # gloss_to_idx = {g: i for i, g in enumerate(unique_glosses)}
# # final_train = augment.create_final_ASL_df(few_classes_train, gloss_to_idx)
# # print(train_asl_df.head())
# # for row in range(9, 15):
# #     book = full_train_asl_df.iloc[row]
# #     gloss = book['ASL_GLOSSES']
# #     videos = book['ASL_VIDS']
# #     video_path = f'archive/videos/{videos}.mp4'
# #     poses_full = book['ASL_POSES']
# #     hands_full = book['ASL_HANDS']

# #     poses_skipped, hands_skipped = augment.skip_frames(poses_full, 2), augment.skip_frames(hands_full, 2)
# #     poses_added, hands_added = augment.add_frames(poses_full), augment.add_frames(hands_full)


# #     for (pose, hand) in [(poses_full, hands_full), (poses_skipped, hands_skipped), (poses_added, hands_added)]:
# #         poses = augment.resample_frames(pose, 15)
# #         hands = augment.resample_frames(hand, 15)
# #         # print(gloss)
# #         # print(videos)
# #         # print(poses)
# #         # print(hands)

# #         cap = cv2.VideoCapture(video_path)

# #         mp_drawing = mp.solutions.drawing_utils
# #         mp_holistic = mp.solutions.holistic

# #         POSE_CONNECTIONS = mp_holistic.POSE_CONNECTIONS
# #         HAND_CONNECTIONS = mp_holistic.HAND_CONNECTIONS

# #         from mediapipe.framework.formats import landmark_pb2

# #         def draw_landmarks(frame, landmarks, connections):
# #             landmark_list = landmark_pb2.NormalizedLandmarkList(
# #                 landmark=[
# #                     landmark_pb2.NormalizedLandmark(x=float(x), y=float(y), z=float(z))
# #                     for x, y, z, v in landmarks
# #                 ]
# #             )

# #             mp_drawing.draw_landmarks(
# #                 frame,
# #                 landmark_list,
# #                 connections
# #             )

# #         frame_idx = 0

# #         while cap.isOpened():
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break

# #             if frame_idx >= len(poses):
# #                 break

# #             pose_frame = poses[frame_idx]

# #             # if hands are combined, adjust indexing accordingly
# #             hand_frame = hands[frame_idx]

# #             # draw pose
# #             draw_landmarks(frame, pose_frame, POSE_CONNECTIONS)

# #             # draw hands
# #             draw_landmarks(frame, hand_frame, HAND_CONNECTIONS)

# #             cv2.imshow("ASL Overlay", frame)
# #             cv2.waitKey(100)
# #             if cv2.waitKey(1) & 0xFF == ord('q'):
# #                 break

# #             frame_idx += 1

# #         cap.release()
# #         cv2.destroyAllWindows()











# # coords = train_asl_df['ASL_COORDS'][0]

# # pose = coords[:33]
# # lh = coords[33:54]
# # rh = coords[54:75]

# # plt.figure(figsize=(5,5))

# # plt.scatter(pose[:,0], pose[:,1], label="pose")
# # plt.scatter(lh[:,0], lh[:,1], label="left hand")
# # plt.scatter(rh[:,0], rh[:,1], label="right hand")

# # plt.gca().invert_yaxis()
# # plt.legend()
# # plt.show()

# # test_asl_df = pd.read_pickle("ASL_dataset_test.pkl")
# # few_classes_test = test_asl_df[test_asl_df['ASL_GLOSSES'].isin(unique_glosses)]
# # unique_glosses = sorted(train_asl_df['ASL_GLOSSES'].unique())
# # gloss_to_idx = {g: i for i, g in enumerate(unique_glosses)}


# # print(few_classes_test)
# # final_train = augment.create_final_ASL_df(train_asl_df, gloss_to_idx)
# # print(np.unique(final_train['ASL_GLOSSES'], return_counts = True))
# # counts = {}

# # import numpy as np
# # import matplotlib.pyplot as plt

# # zero_ratio = final_train['ASL_COORDS'].apply(
# #     lambda entry: np.mean(
# #         np.all(np.array(entry) == 0, axis=-1)
# #     )
# # )

# # plt.hist(zero_ratio, bins=50)
# # plt.title("Distribution of zero-valued coordinates per sample")
# # plt.xlabel("Number of zeros")
# # plt.ylabel("Frequency")
# # plt.show()
# # for i, row in final_train.iterrows():
# #     entry = row['ASL_COORDS']
# #     count = 0
# #     for frame in entry:
# #         for coord in frame:
# #             if coord == np.zeros(4):
# #                 count+=1
# #     counts[i] = count
# # print(counts)


# # final_test = augment.create_final_ASL_df(test_asl_df, gloss_to_idx)
# # final_test.to_pickle("FULL_ASL_dataset_test.pkl")
# # print(len(final_test), len(test_asl_df['ASL_GLOSSES'].unique()))


# # train = train_asl_df[:30]
# # train_glosses = set(train['ASL_GLOSSES'])
# # test = test_asl_df[test_asl_df['ASL_GLOSSES'].isin(train_glosses)]
# # print(len(np.unique(test_asl_df['ASL_GLOSSES'])))

# # print(np.unique(train_asl_df['ASL_GLOSSES']))
# # print(len(np.unique(t_asl_df['ASL_GLOSSES'])))

# # train_cousin = train_asl_df[train_asl_df['ASL_GLOSSES']=="cousin"]
# # test_cousin = test_asl_df[test_asl_df['ASL_GLOSSES']=="cousin"]

# # test_cousin_pose = test_cousin['ASL_POSES'].iloc[0]
# # train_cousin_pose = train_cousin['ASL_POSES'].iloc[0]
# # train_cousin_pose = train_cousin_pose[4]
# # test_cousin_pose = test_cousin_pose[1]
# # test_left_shoulder = test_cousin_pose[:, 11, :]
# # test_right_shoulder = test_cousin_pose[:, 12, :]
# # test_left_hip = test_cousin_pose[:, 23, :]
# # test_right_hip = test_cousin_pose[:, 24, :]
# # test_ref = (test_left_shoulder + test_right_shoulder + test_left_hip + test_right_hip) / 4
# # test_pose = test_cousin_pose - test_ref[:, None, :]

# # train_left_shoulder = train_cousin_pose[:, 11, :]
# # train_right_shoulder = train_cousin_pose[:, 12, :]
# # train_left_hip = train_cousin_pose[:, 23, :]
# # train_right_hip = train_cousin_pose[:, 24, :]
# # train_ref = (train_left_shoulder + train_right_shoulder + train_left_hip + train_right_hip) / 4
# # train_pose = train_cousin_pose - train_ref[:, None, :]

# # def resample_frames(landmarks, max_frames):
# #     T = len(landmarks)
# #     if T == max_frames:
# #         return landmarks
# #     indices = np.linspace(0, T - 1, max_frames).astype(int)
# #     return landmarks[indices]

# # train_resampled = resample_frames(train_pose, 30)
# # test_resampled = resample_frames(test_pose, 30)

# # for i in range(len(train_resampled)):
# #     for j in range(32):
# #         print(train_resampled[i][j])
# #         print(test_resampled[i][j])
# #         print('----------')

# # # for i in range(len(test_cousin_pose[0])):
# # #     print(test_cousin_pose[0][i])

# # # # train_cousin_pose = (train_cousin['ASL_POSES'].iloc[0])
# # # # for i in range(len(train_cousin_pose[0])):
# # # #     print(train_cousin_pose[0][i])

# # # # print(train_cousin_pose[0][0][0])
# # # # print(test_cousin_pose[0][0][0])

# # # for i in range(15):
# # #     print(train_cousin_pose[0][i][0] - test_cousin_pose[0][i][0])

