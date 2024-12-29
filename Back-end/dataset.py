from torch.utils.data import DataLoader, Dataset
import cv2
from skimage import color
import torch


class VideoDataset(Dataset):
    def __init__(self, input_video_path, transform=None):
        self.input_video_path = input_video_path
        self.transform = transform
        self.frames = self.load_video_frames()
        self.size = len(self.frames)
    
    def __getitem__(self, index):
        # Load the frame and convert to RGB (OpenCV loads in BGR)
        frame = self.frames[index][:, :, ::-1].copy()  # Ensure a copy of the array is made
        frame_tensor = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
        lab_image = rgb_to_lab(frame_tensor)
        L, a, b = lab_image
        L = L.unsqueeze(dim=0)
        return L
    
    def __len__(self):
        return self.size

    def load_video_frames(self):
        cap = cv2.VideoCapture(self.input_video_path)

        if not cap.isOpened():
            raise IOError(f"Error opening video file: {self.input_video_path}")
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

def rgb_to_lab(image_data):
    # Convert tensor to numpy for skimage processing
    image = image_data.permute(1, 2, 0).numpy()  # Convert (C, H, W) to (H, W, C)
    lab_image = color.rgb2lab(image)  # Convert RGB to LAB
    lab_image = torch.tensor(lab_image).permute(2, 0, 1)  # Convert back to (C, H, W)
    return lab_image
