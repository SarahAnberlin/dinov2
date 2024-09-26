import os
from PIL import Image
from torchvision import transforms

from hubconf import dinov2_vits14

img_root = f'/dataset/vfayezzhang/dataset/MiniImageNet1k'

files = [f for f in os.listdir(img_root) if f.endswith('.jpg') or f.endswith('.png')]
file_paths = [os.path.join(img_root, f) for f in files]

model = dinov2_vits14()
for file_path in file_paths:
    image = Image.open(file_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    image = transform(image).unsqueeze(0)
    feature = model(image, is_training=True)
    print(f"Type of feature: {type(feature)}")
