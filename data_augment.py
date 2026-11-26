import numpy as np
import pandas as pd
#https://www.mdpi.com/2076-3417/11/23/11481#sec3-applsci-11-11481

def find_max_frames(asl_df):
    max_frames = 0
    for i, row in asl_df.iterrows():
        for j in range(len(row['ASL_POSES'])):
            if len(row['ASL_POSES'][j]) > max_frames:
                max_frames = len(row['ASL_POSES'][j])
    return max_frames

# def resample_frames(landmarks, max_frames):
#     T = len(landmarks)
#     if T == max_frames:
#         return landmarks
#     indices = np.linspace(0, T - 1, max_frames).astype(int)
#     return landmarks[indices]


def resample_frames(landmarks, max_frames):
    T = len(landmarks)
    if T == max_frames:
        return landmarks

    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, max_frames)

    resampled = np.zeros((max_frames, *landmarks.shape[1:]))

    for j in range(landmarks.shape[1]):      # joints
        for k in range(landmarks.shape[2]):  # xyz
            resampled[:, j, k] = np.interp(new_idx, old_idx, landmarks[:, j, k])

    return resampled

# Skip frames
def skip_frames(landmarks, skip_frame):
    landmarks_skipped = np.zeros_like(landmarks)
    landmarks_skipped[0] = landmarks[0]
    landmarks_skipped[-1] = landmarks[-1]
    for i in range(1, len(landmarks)-(skip_frame + 1)):
        if i % skip_frame == 0:
            landmarks_skipped[i] = (landmarks[i-1] + landmarks[i+1])/2
        else:
            landmarks_skipped[i] = landmarks[i]
    return landmarks_skipped

# Add frames
def add_frames(landmarks):
    landmarks_added = np.zeros((landmarks.shape[0]*2, landmarks.shape[1], landmarks.shape[2]))
    add_frame = 2
    landmarks_added[0] = landmarks[0]
    landmarks_added[-1] = landmarks[-1]
    for i in range(1, len(landmarks)-1):
        if i % add_frame == 0:
            landmarks_added[i] = (landmarks[i-1] + landmarks[i])/2
        else:
            landmarks_added[i] = landmarks[i]
    return landmarks_added

def mirror_coords(landmarks):
    landmarks_mirrored = landmarks.copy()
    landmarks_mirrored[:, :, 0] = 1 - landmarks[:, :, 0]
    return landmarks_mirrored

def add_gaussian_noise(landmarks, std=0.01):
    noise = np.random.normal(0, std, landmarks.shape)
    return landmarks + noise

def rotate_around_point(joints, angle_x, angle_y, angle_z, origin=None):
    angle_x, angle_y, angle_z = np.radians(angle_x), np.radians(angle_y), np.radians(angle_z)
    cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
    cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
    cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
    
    R_x = np.array([[1, 0, 0],
                    [0, cos_x, -sin_x],
                    [0, sin_x,  cos_x]])
    R_y = np.array([[cos_y, 0, sin_y],
                    [ 0, 1, 0],
                    [-sin_y, 0, cos_y]])
    R_z = np.array([[cos_z, -sin_z, 0],
                    [sin_z,  cos_z, 0],
                    [0,  0, 1]])
    R = R_z @ R_y @ R_x
    
    if origin is not None:
        joints = joints - origin   # translate to origin
    joints = joints @ R.T          # apply rotation
    if origin is not None:
        joints = joints + origin   # translate back
    return joints


def augment_hands(hands, angle_x, angle_y, angle_z):
    angle_x, angle_y, angle_z = np.random.uniform(-angle_x, angle_x), np.random.uniform(-angle_y, angle_y), np.random.uniform(-angle_z, angle_z)
    hand_1, hand_2 = hands[:, :21, :], hands[:, 21:, :]
    wrist_1, wrist_2 = hand_1[:, 0:1, :], hand_2[:, 0:1, :]    # (frames, 1, 3) — use wrist as pivot
    hand_1 = rotate_around_point(hand_1, angle_x, angle_y, angle_z, origin=wrist_1)
    hand_2 = rotate_around_point(hand_2, angle_x, angle_y, angle_z, origin=wrist_2)
    hands = np.concatenate([hand_1, hand_2], axis=1)
    return hands

def flatten_landmarks(pose, hands, max_frames, model_type):
    # 11, 23, 17
    max_frames = 30
    # remove = {2, 3, 6, 7, 10, 11, 14, 15, 18, 19}
    # remove = {9, 10}
    # remove = {15, 16, 17, 18, 19, 20, 21, 22} #no hands from pose
    # remove = {16, 14, 13, 15} #no forearms from pose
    # remove = {16, 14, 13, 15} #no forearms from pose
    # remove = {12, 14, 13, 11} #no upper arms from pose
    # remove = {0, 1, 2, 3, 21, 22, 23, 24} #no thumb
    # remove = {5, 6, 7, 8, 26, 27, 28, 29} #no index
    # remove = {9, 10, 11, 12, 30, 31, 32, 33} #no middle
    # remove = {13, 14, 15, 16, 34, 35, 36, 37} #no ring finger
    remove = {17, 18, 19, 20, 38, 39, 40, 41} #no thumb
    # keep = [i for i in range(pose.shape[1]) if i not in remove]
    keep = [i for i in range(hands.shape[1]) if i not in remove]
    pose = pose[:, :, 0:3]
    hands = hands[:, :, 0:3]

    combined = np.concatenate((pose, hands), axis=1)
    combined_resample = combined
    combined_resample = resample_frames(combined, max_frames)
    combined_resample = np.nan_to_num(combined_resample, nan=0.0)
    
    if combined.shape[0] < max_frames:
        padding = np.zeros((max_frames - combined.shape[0], combined.shape[1], combined.shape[2]))  # zeros for remaining frames
        combined = np.vstack([combined, padding])
    if model_type == 'MLP':
        flattened = combined_resample.flatten()
    elif model_type == 'T' or model_type == 'L':
        flattened = combined_resample.reshape(combined_resample.shape[0], -1)
        # flattened = hands.reshape(hands.shape[0], -1)
    elif model_type == 'S':
        flattened = combined_resample
        flattened = np.transpose(flattened, (2, 0, 1))
    return flattened


# def create_final_ASL_df(asl_df, gloss_to_idx):
#     asl_coords = []
#     asl_glosses = []
#     max_frames = 30
#     for i, row in asl_df.iterrows():
#         gloss = row['ASL_GLOSSES']
#         if gloss not in gloss_to_idx:
#                 raise ValueError(f"Unknown gloss: {gloss}")
#         label_id = gloss_to_idx[gloss]
#         for j in range(len(row['ASL_POSES'])):
#             pose, hands = np.array(row['ASL_POSES'][j]), np.array(row['ASL_HANDS'][j])
#             pose = pose[:, :, 0:2]
#             hands = hands[:, :, 0:2]
#             left_shoulder = pose[:, 11, :]
#             right_shoulder = pose[:, 12, :]
#             left_hip = pose[:, 23, :]
#             right_hip = pose[:, 24, :]
#             ref = (left_shoulder + right_shoulder + left_hip + right_hip) / 4
#             pose = pose - ref[:, None, :]
#             hands = hands - ref[:, None, :]
#             # Regular
#             flattened = flatten_landmarks(pose, hands, max_frames)
#             # flattened = augment.resample_frames(flattened, max_frames)
#             asl_glosses.append(label_id)
#             asl_coords.append(flattened)
#             # Skip frames
#             for k in [2,3,4,5]:
#                 pose_skipped, hands_skipped = skip_frames(pose, k), skip_frames(hands, k)
#                 flattened_skipped = flatten_landmarks(pose_skipped, hands_skipped, max_frames)
#                 asl_glosses.append(label_id)
#                 asl_coords.append(flattened_skipped)
#             # Add frames
#             pose_added, hands_added = add_frames(pose), add_frames(hands)
#             flattened_added = flatten_landmarks(pose_added, hands_added, max_frames)
#             asl_glosses.append(label_id)
#             asl_coords.append(flattened_added)
#     asl_final_df = pd.DataFrame(data = {"ASL_GLOSSES":asl_glosses, "ASL_COORDS":asl_coords})
#     return asl_final_df


