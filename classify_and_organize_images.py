import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import shutil
import matplotlib.pyplot as plt

# Setup the Model
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # Adjust according to your number of classes
model.load_state_dict(torch.load('models/model_trained.pth'))
model.eval()  # Set the model to inference mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create a Custom Dataset
class FolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, img_name

# Check and create output and class folders
output_dir = 'output'
class_names = ['Memes', 'Paisajes', 'Personas', 'Varios']  # Adjust as per your classes
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for class_name in class_names:
    class_dir = os.path.join(output_dir, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

# Load Images and Create DataLoader
predict_dir = 'data_predict'
predict_dataset = FolderDataset(root_dir=predict_dir, transform=transform)
predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False)  # Batch size set to 1 for individual processing

# Classify and copy images to corresponding class folders
for images, paths in predict_loader:
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    predicted_class = class_names[preds[0]]
    destination_folder = os.path.join(output_dir, predicted_class)
    for path in paths:
        shutil.copy(path, destination_folder)

print("Classification and organization of images completed.")
