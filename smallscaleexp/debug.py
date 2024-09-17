from PIL import Image, ImageOps
import torch
from torchvision import transforms

# Load a sample image (replace with your own image path)
image_path = 'th.jpg'
img = Image.open(image_path)

# Define the transformation (including Solarize)
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Adjust parameters if needed
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.ToPILImage(),
  # Convert back to PIL Image for solarize
    transforms.Lambda(lambda img: ImageOps.solarize(img, threshold=128))  # Apply solarize
])

# Apply the transformation
try:
    transformed_img = transform(img)
    print("Transformation successful!")
except Exception as e:
    print(f"Error during transformation: {e}")