import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk  # Import ttk module for themed widgets

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

def classify_images(source_dir, output_dir, model, device, transform, class_names):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for class_name in class_names:
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    predict_dataset = FolderDataset(root_dir=source_dir, transform=transform)
    predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False)

    for images, paths in predict_loader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds[0]]
        destination_folder = os.path.join(output_dir, predicted_class)
        for path in paths:
            shutil.copy(path, destination_folder)
    messagebox.showinfo("Success", "Images have been classified and organized.")

def main():
    root = tk.Tk()
    root.title("Image Classifier")
    root.geometry("400x200")  # Adjust the size of the window as needed
    
    style = ttk.Style()
    style.theme_use('clam')  # You can experiment with other themes: 'alt', 'default', 'classic', 'clam'

    def select_source_folder():
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            source_entry.delete(0, tk.END)
            source_entry.insert(0, folder_selected)

    def select_output_folder():
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            output_entry.delete(0, tk.END)
            output_entry.insert(0, folder_selected)

    def start_classification():
        source_dir = source_entry.get()
        output_dir = output_entry.get()
        if not source_dir or not output_dir:
            messagebox.showerror("Error", "Please select both source and destination folders.")
            return
        classify_images(source_dir, output_dir, model, device, transform, class_names)

    ttk.Label(root, text="Source Folder:").pack()
    source_entry = ttk.Entry(root, width=50)
    source_entry.pack()
    ttk.Button(root, text="Browse", command=select_source_folder).pack()

    ttk.Label(root, text="Output Folder:").pack()
    output_entry = ttk.Entry(root, width=50)
    output_entry.pack()
    ttk.Button(root, text="Browse", command=select_output_folder).pack()

    ttk.Button(root, text="Classify Images", command=start_classification).pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
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

    class_names = ['Memes', 'Paisajes', 'Personas', 'Varios']  # Adjust as per your classes
    main()
