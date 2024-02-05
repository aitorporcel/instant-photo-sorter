import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

# Step 1: Setup the Model
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # Adjust according to your number of classes
model.load_state_dict(torch.load('models/model_trained.pth'))
model.eval()  # Set the model to inference mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Step 2: Define the Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Step 3: Create a Custom Dataset
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
        return image

# Step 4: Load Images and Create DataLoader
predict_dir = 'data_predict'
predict_dataset = FolderDataset(root_dir=predict_dir, transform=transform)
predict_loader = DataLoader(predict_dataset, batch_size=16, shuffle=False)

# Step 5: Predict in Batches and Visualize
class_names = ['Memes', 'Paisajes', 'Personas', 'Varios']  # Adjust as per your classes

def visualize_predictions(images, preds, class_names):
    plt.figure(figsize=(12, 12))
    batch_size = len(images)
    for i in range(batch_size):
        ax = plt.subplot(batch_size // 4 + 1, 4, i + 1)
        img = images[i].permute(1, 2, 0).numpy()
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Unnormalize
        plt.imshow(img)
        plt.title(class_names[preds[i]])
        plt.axis("off")
    plt.show()

for images in predict_loader:
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    visualize_predictions(images.cpu(), preds, class_names)
