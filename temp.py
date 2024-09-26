import os
from PIL import Image
from torchvision import transforms
import torch

from hubconf import dinov2_vits14, dinov2_vitl14, dinov2_vitb14

img_root = f'/dataset/vfayezzhang/dataset/MiniImageNet1k'

files = [f for f in os.listdir(img_root) if f.endswith('.jpg') or f.endswith('.png')]
file_paths = [os.path.join(img_root, f) for f in files]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = dinov2_vitl14().to(device).eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((490, 742)),
])

for file_path in file_paths:
    image = Image.open(file_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    feature = model(image, is_training=True)
    # The feature is a dict, print the keys to see what's inside
    print(f"Feature keys: {feature.keys()}")
    for key, value in feature.items():
        print(f"Feature {key}: {value.shape}")
