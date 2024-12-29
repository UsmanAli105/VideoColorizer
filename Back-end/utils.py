import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


def save_net(net, path, file_name):
    """
    Save the state dictionary of a PyTorch model to a file.

    This function saves the parameters of the given PyTorch model to a specified file path.
    If the directory for the path does not exist, it will be created.

    Args:
        net (torch.nn.Module): The PyTorch model instance whose state dictionary is to be saved.
        path (str): The directory path where the model file will be saved.
        file_name (str): The name of the file to save the model's state dictionary (without the '.pth' extension).

    Returns:
        None
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    torch.save(net.state_dict(), os.path.join(path, f"{file_name}.pth"))


def load_model(saved_models_folder_path, saved_model_name, device):
    """
    Load the state dictionary of a PyTorch model from a checkpoint file.

    This function loads the state dictionary from a specified file path and returns it. The state dictionary contains
    the model's parameters.

    Args:
        saved_models_folder_path (str): The directory path where the model checkpoint file is located.
        saved_model_name (str): The name of the checkpoint file (without the '.pth' extension).
        device(str): CPU / GPU

    Returns:
        dict: The state dictionary of the PyTorch model.
    """
    checkpoint_path = os.path.join(saved_models_folder_path, f"{saved_model_name}.pth")
    state_dict = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=True)
    return state_dict


def display_loss(train_loss, val_loss):
    num_epoch = np.linspace(0, len(train_loss), len(train_loss))
    plt.title('Loss graph')
    plt.plot(num_epoch, train_loss, label='train')
    plt.plot(num_epoch, val_loss, label='validation')
    plt.legend()
    plt.show()


def split_dataset(dataset, val_split=0.2):
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=val_split, random_state=42)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


def show_sample_images(dataloader, class_names):
    """
    Displays a 2x2 grid of images from the DataLoader with their labels as captions.
    
    Args:
        dataloader (DataLoader): DataLoader object containing the dataset.
        class_names (list): List of class names for labeling the images.
    """
    # Get a batch of images from the dataloader
    images, labels = next(iter(dataloader))

    # Convert tensors to numpy arrays for visualization
    images = images.numpy()
    
    # Create a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    
    # Display 4 images with their labels
    for i, ax in enumerate(axs.flatten()):
        img = np.transpose(images[i], (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        ax.imshow(img)
        ax.set_title(f"Label: {class_names[labels[i]]}")
        ax.axis('off')  # Hide axes for cleaner visualization
    
    plt.tight_layout()
    plt.show()

def visualize_single_image(image_data):
    """
    Visualizes a single image of shape (3, 256, 256) where the pixel values are floats.
    
    Args:
        image_data (Tensor): A float tensor of shape (3, 256, 256).
    """
    if isinstance(image_data, torch.Tensor):
        image_data = image_data.numpy()  # Convert to numpy if it's a torch tensor
    
    # Convert from (C, H, W) to (H, W, C) for visualization
    img = np.transpose(image_data, (1, 2, 0))
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')  # Hide axes for cleaner visualization
    plt.show()

def visualize_two_images(image1, image2):
    """
    Visualizes two images side by side in a 1x2 grid.
    
    Args:
        image1 (Tensor): First image to display.
        image2 (Tensor): Second image to display.
    """
    if isinstance(image1, torch.Tensor):
        image1 = image1.numpy()  # Convert to numpy if it's a torch tensor
    if isinstance(image2, torch.Tensor):
        image2 = image2.numpy()  # Convert to numpy if it's a torch tensor

    # Convert from (C, H, W) to (H, W, C) for both images
    img1 = np.transpose(image1, (1, 2, 0))
    img2 = np.transpose(image2, (1, 2, 0))
    
    # Create a 1x2 grid to display the two images
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    axs[0].imshow(img1)
    axs[0].axis('off')  # Hide axes for cleaner visualization
    
    axs[1].imshow(img2)
    axs[1].axis('off')  # Hide axes for cleaner visualization
    
    plt.tight_layout()
    plt.show()
