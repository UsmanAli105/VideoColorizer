from torchvision.io import read_image
import matplotlib.pyplot as plt
from skimage import color
from tqdm import tqdm
from dataset import *
from models import *
import numpy as np
from utils import *
import os
import imageio
import torch
import cv2


def load_image(image_path:str):
    if not os.path.exists(image_path):
        raise Exception(f'Provided path {image_path} is invalid.')
    image_data = read_image(image_path).float() / 255.0
    return image_data


def display_image(image_data):
    image = image_data.permute(1, 2, 0).numpy()
    plt.title('Sample Image')
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def rgb_to_lab(image_data):
    image = image_data.permute(1, 2, 0).numpy()
    lab_image = color.rgb2lab(image)
    lab_image = torch.tensor(lab_image).permute(2, 0, 1)
    return lab_image

def lab_to_rgb(image_data):
    batch_size = image_data.shape[0]
    rgb_images = []
    for i in range(batch_size):
        image = image_data[i].permute(1, 2, 0).numpy()
        rgb_image = color.lab2rgb(image)
        rgb_image = torch.tensor(rgb_image).permute(2, 0, 1)
        rgb_images.append(rgb_image)
    return torch.stack(rgb_images)


def colorize_frames(eccv, siggraph, images, alpha=0.1, beta=0.9):
    output_eccv = eccv(images)
    output_siggraph = siggraph(images)

    combine_output = (alpha * output_eccv + beta * output_siggraph)

    ab_channel = combine_output.cpu()
    images = images.cpu()

    lab_image = torch.cat((images, ab_channel), dim=1)
    colored_images = lab_to_rgb(lab_image)
    return colored_images
    

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def process_and_export_video(eccv, siggraph, data_loader, colorize_frames, output_video_path, device, fps=30, max_frames=1600):
    with imageio.get_writer(output_video_path, fps=fps, codec='libx264') as writer:
        count = 0
        with torch.no_grad():
            for images in tqdm(data_loader, desc="Processing frames"):
                images = images.to(device)
                colored_frame = colorize_frames(eccv, siggraph, images)
                images = images.to('cpu')
                frame = (colored_frame.permute(0, 2, 3, 1).squeeze(0) * 255.0).cpu().numpy().astype(np.uint8)
                writer.append_data(frame)
                torch.cuda.empty_cache()
                if count == max_frames:
                    break
                count += 1
    print(f"Video saved to {output_video_path}")


def main_colorize():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    eccv = ECCVGenerator()
    siggraph = SIGGRAPHGenerator()

    eccv.to(device=device)
    siggraph.to(device=device)

    weights_dir = 'weights'
    eccv_weights_path = 'trained_eccv_weights'
    siggraph_weights_path = 'trained_siggraph_weights'

    eccv_weights = load_model(weights_dir, eccv_weights_path, device)
    siggraph_weights = load_model(weights_dir, siggraph_weights_path, device)

    eccv.load_state_dict(eccv_weights)
    siggraph.load_state_dict(siggraph_weights)

    eccv.eval()
    siggraph.eval()

    dataset = VideoDataset('uploads/temp.mp4')
    data_loader = DataLoader(dataset, batch_size=1)
    process_and_export_video(eccv, siggraph, data_loader, colorize_frames, "output/output.mp4", device=device, fps=23, max_frames=9999)