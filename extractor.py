import os
os.environ["GLOG_minloglevel"] = "2"   # suppress INFO + WARNING
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # extra suppression
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os
print("IN EXTRACTOR --------------------")

def pose_extractor(video_id, sl):
    model_path = 'pose_landmarker_lite.task'
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO)
    landmarks_list = []
    if sl == "asl":
        video_path = f'archive/videos/{video_id}.mp4'
    elif sl == "isl":
        video_path = video_id
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        timestamp_ms = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # fallback
        frame_duration_ms = int(1000 / fps)
        with PoseLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb
                )

                results = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks[0]
                    frame_landmarks = [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]
                    landmarks_list.append(frame_landmarks)
                    # landmarks is a list of 33 PoseLandmark objects
                    # each has x, y, z, visibility
                else:
                    frame_landmarks = np.zeros((33, 4), dtype=np.float32)
                    landmarks_list.append(frame_landmarks)
                timestamp_ms += frame_duration_ms
            # np.save(f"skeletons/{video_name[:-4]}_landmarks.npy",np.array(landmarks_list, dtype=np.float32))
            cap.release()
            return np.array(landmarks_list, dtype=np.float32)
    else:
        return None

def hand_extractor(video_id, sl):
    model_path = 'hand_landmarker.task'
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )

    landmarks_list = []
    if sl == "asl":
        video_path = f'archive/videos/{video_id}.mp4'
    elif sl == "isl":
        video_path = video_id
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        timestamp_ms = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30 
        frame_duration_ms = int(1000 / fps)
        with HandLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb
                )
                results = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )
                # frame_landmarks = np.array([])
                # if results.hand_landmarks:
                #     print(len(results.hand_landmarks))
                #     for h in range(len(results.hand_landmarks)):
                #         hand_landmarks = results.hand_landmarks[h]
                #         one_hand_landmarks = [np.array([lm.x, lm.y, lm.z, lm.visibility]) for lm in hand_landmarks]
                #         if len(frame_landmarks) == 0:
                #             frame_landmarks = one_hand_landmarks
                #         else:
                #             frame_landmarks = np.concatenate([frame_landmarks, one_hand_landmarks], axis = 1)
                #     while len(frame_landmarks[0]) < 42:
                #         frame_landmarks = np.concatenate([frame_landmarks, np.zeros((21, 4))], axis = 1)
                # else:
                #     frame_landmarks = [np.zeros((42, 4))]
                # landmarks_list.append(frame_landmarks)
                frame_landmarks = np.zeros((42, 4))  # always fixed size

                if results.hand_landmarks:
                    for h, hand_landmarks in enumerate(results.hand_landmarks[:2]):  # max 2 hands
                        one_hand = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in hand_landmarks])
                        start = h * 21
                        frame_landmarks[start:start+21] = one_hand

                landmarks_list.append(frame_landmarks)

                timestamp_ms += frame_duration_ms
            # np.save(f"hand_skeletons/{video_name[:-4]}_hand_landmarks.npy",np.array(landmarks_list, dtype=np.float32))
            cap.release()
            # for ele in landmarks_list:
            #     if len(ele[0]) != 42:
            #         print(len(ele[0]))
            #         print(ele)
            return np.array(landmarks_list, dtype=np.float32)
    else:
        return None
