import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import transforms
from deepface import DeepFace
import pickle

REFRESH_KNOWN_FACE = False
SOURCE_VIDEO = "./source/VID_20240131_181322.mp4"  # 输入视频
KNOWN_FACES_DIR = "./known_pic"
FONT_PATH = "./fonts/Noto_Sans_SC/static/NotoSansSC-Regular.ttf"
OUTPUT_PATH = "result/output_video.mp4" # 输出视频
SKIP_FRAMES = 100

# 设置字体
font_path = FONT_PATH  # 字体文件路径
font_size = 20
font = ImageFont.truetype(font_path, font_size)

# 初始化facenet用于检测和特征提取
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# 视频处理
video_capture = cv2.VideoCapture(SOURCE_VIDEO)
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
fps = video_capture.get(cv2.CAP_PROP_FPS)
print(fps)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

frame_count = 0
skip_frames = SKIP_FRAMES

if REFRESH_KNOWN_FACE is True:
    # 加载已知人脸库
    known_face_encodings = []
    known_face_names = []

    # 假设您有一个目录包含已知人脸图片
    known_faces_dir = KNOWN_FACES_DIR
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg"):
            image = Image.open(os.path.join(known_faces_dir, filename))
            # MTCNN 检测面部
            face = mtcnn(image)
            # 确保检测到面部，并且维度正确
            if face is not None:
                # 如果 face 是 3D 张量，增加一个批处理维度
                if face.dim() == 3:
                    face = face.unsqueeze(0)
                # 获取面部编码
                encoding = resnet(face)
                # 保存编码和名字
                known_face_encodings.append(encoding)
                print(filename[:-4])
                known_face_names.append(filename[:-4])
                

    # 将人脸编码和名字保存到文件
    data = {"encodings": known_face_encodings, "names": known_face_names}
    with open("known_faces.pkl", "wb") as file:
        pickle.dump(data, file)

    print("已知人脸数据保存完毕。")

else:
    # 加载已知人脸数据
    with open("known_faces.pkl", "rb") as file:
        data = pickle.load(file)
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]

    print("已知人脸数据加载完毕。")


while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break
    
    frame_count += 1
    print(frame_count)
    if frame_count >= 5000 or frame_count % skip_frames != 0:
        continue

    frame_height, frame_width, frame_channels = frame.shape
    # 将OpenCV的BGR图像转换为RGB图像
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 检测和识别人脸
    frame_pil = Image.fromarray(frame_rgb)
    boxes, _ = mtcnn.detect(frame_pil)
    faces = mtcnn(frame_pil)
    
    draw = ImageDraw.Draw(frame_pil)
    
    if boxes is not None:
        for box, face in zip(boxes, faces):
            if face is None:
                print("face is None")
                continue
            
            left, top, right, bottom = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            # 画框
            draw.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0), width=2)
            
            min_dist = 0.70
            # 比较已知人脸
            face_encoding = resnet(face.unsqueeze(0))
            distances = [(face_encoding - known_face).norm().item() for known_face in known_face_encodings]
            best_match_index = np.argmin(distances)
            if distances[best_match_index] < min_dist:  # 阈值设定
                name = known_face_names[best_match_index]
            else:
                print("UNKNOWN PERSON")
                name = ""
            
            # 画标签
            draw.text((left, top - 20), f"{name}", font=font, fill=(255, 255, 0))

    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    out.write(frame)

video_capture.release()
out.release()
cv2.destroyAllWindows()



