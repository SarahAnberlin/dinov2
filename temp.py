import os
from PIL import Image
from torchvision import transforms
import torch

from hubconf import dinov2_vits14

img_root = f'/dataset/vfayezzhang/dataset/MiniImageNet1k'

files = [f for f in os.listdir(img_root) if f.endswith('.jpg') or f.endswith('.png')]
file_paths = [os.path.join(img_root, f) for f in files]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = dinov2_vits14().to(device).eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

for file_path in file_paths:
    image = Image.open(file_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    feature = model(image, is_training=True)
    print(f"Type of feature: {type(feature)}")
