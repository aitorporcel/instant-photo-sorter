import sys
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import os
import shutil
from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit, QProgressBar, QMessageBox
from PySide2.QtGui import QIcon, QFont
from PySide2.QtCore import Qt, QThread, Signal
from datetime import datetime

import sys
from pathlib import Path
import torch

# Function to get the path of the data directory
def resource_path(relative_path):
    """ Get the absolute path to the resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = Path(__file__).parent

    return Path(base_path) / relative_path


class FolderDataset(Dataset):
    """
    Dataset to load images from a folder.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
        self.failed_images = []  # Keep track of failed images

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        try:
            image = Image.open(img_name)
            if self.transform:
                image = self.transform(image)
        except UnidentifiedImageError:
            self.failed_images.append(img_name)
            # Instead of returning None, return a placeholder tensor and the image name
            placeholder = torch.zeros(1, 1, 1)  # Adjust dimensions as needed
            return placeholder, img_name
        return image, img_name


def classify_images(source_dir, output_dir, model, device, transform, class_names, progress_callback, failed_folder="Failed"):
    """Classify images from source_dir using model and output to output_dir.

    Args:
        source_dir (str): Directory containing images to classify.
        output_dir (str): Directory to output classified images.
        model (torch.nn.Module): PyTorch model to use for classification.
        device (torch.device): Device to run model on.
        transform (torchvision.transforms): Transforms to apply to images before classifying.
        class_names (List[str]): List of class name strings.
        progress_callback (Callable): Callback to report progress.
        failed_folder (str): Folder name for failed classifications.

    """
    
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure failed folder exists
    for class_name in class_names + [failed_folder]:  
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    # Classify images
    predict_dataset = FolderDataset(root_dir=source_dir, transform=transform)
    predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False)

    dataset_size = len(predict_dataset)
    for i, (images, paths) in enumerate(predict_loader):
        
        # Check for placeholder tensor indicating failure
        if isinstance(images, torch.Tensor) and images.nelement() == 1:  
            predicted_class = failed_folder
        
        else:
            # Classify image
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
            predicted_class = class_names[preds[0]]
        
        # Copy image to output folder
        destination_folder = os.path.join(output_dir, predicted_class)
        for path in paths:
            shutil.copy(path, destination_folder)

        # Emit progress update
        progress = int((i + 1) / dataset_size * 100)
        progress_callback.emit(progress)


class ClassificationThread(QThread):
    """
    Thread for running the classification in the background.
    """
    update_progress = Signal(int)
    classification_done = Signal(str)  # Modified to emit a string with the elapsed time

    def __init__(self, source_dir, output_dir, model, device, transform, class_names):
        QThread.__init__(self)
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.model = model
        self.device = device
        self.transform = transform
        self.class_names = class_names

    def run(self):
        start_time = datetime.now()  # Start timing
        classify_images(self.source_dir, self.output_dir, self.model, self.device, self.transform, self.class_names, self.update_progress)
        elapsed_time = datetime.now() - start_time  # Calculate elapsed time
        minutes, seconds = divmod(elapsed_time.seconds, 60)
        elapsed_time_str = f"{minutes} minutes, {seconds} seconds"
        self.classification_done.emit(elapsed_time_str)  # Emit the elapsed time

class App(QWidget):
    """
    Main application window.
    """
    def __init__(self, model, device, transform, class_names):
        super().__init__()
        self.title = 'Image Classifier'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.model = model
        self.device = device
        self.transform = transform
        self.class_names = class_names
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Image Classifier")
        self.setGeometry(300, 300, 600, 300)  # Updated for better layout
        self.setFont(QFont("Arial", 10))  # Modern font
        self.setWindowIcon(QIcon("app_icon.png"))  # Set the path to your application icon
        
        layout = QVBoxLayout()
        
        # Source Folder Section
        self.sourceLabel = QLabel("Source Folder:")
        layout.addWidget(self.sourceLabel)
        self.sourceEdit = QLineEdit(self)
        layout.addWidget(self.sourceEdit)
        self.sourceBtn = QPushButton('Browse Source Folder')
        self.sourceBtn.clicked.connect(self.selectSourceFolder)
        layout.addWidget(self.sourceBtn)
        
        # Output Folder Section
        self.outputLabel = QLabel("Output Folder:")
        layout.addWidget(self.outputLabel)
        self.outputEdit = QLineEdit(self)
        layout.addWidget(self.outputEdit)
        self.outputBtn = QPushButton('Browse Output Folder')
        self.outputBtn.clicked.connect(self.selectOutputFolder)
        layout.addWidget(self.outputBtn)
        
        # Add some vertical spacing
        spacer = QLabel("")  # An empty label can act as a spacer
        layout.addWidget(spacer)
        
        # Classify Button with style and spacing
        self.classifyBtn = QPushButton('Classify Images')
        self.classifyBtn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;  /* Blue */
                color: white;
                border-radius: 5px;  /* Rounded corners */
                padding: 5px;  /* Padding inside the button */
                border: 1px solid #0D47A1;  /* Slightly darker blue border for depth */
            }
            QPushButton:hover {
                background-color: #1976D2;  /* Slightly darker blue when mouse hovers over */
            }
            QPushButton:pressed {
                background-color: #0D47A1;  /* Even darker blue when button is pressed */
            }
        """)
        self.classifyBtn.clicked.connect(self.startClassification)
        layout.addWidget(self.classifyBtn)
        
        # Add spacing after the button
        layout.addSpacing(20)
        
        # Progress Bar
        self.progressBar = QProgressBar(self)
        self.progressBar.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progressBar)
        
        self.setLayout(layout)
    
    def selectSourceFolder(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Source Directory")
        if directory:
            self.sourceEdit.setText(directory)
    
    def selectOutputFolder(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.outputEdit.setText(directory)
    
    def startClassification(self):
        source_dir = self.sourceEdit.text()
        output_dir = self.outputEdit.text()
        if source_dir and output_dir:
            self.classification_thread = ClassificationThread(
                source_dir, output_dir, self.model, self.device, self.transform, self.class_names)
            self.classification_thread.update_progress.connect(self.updateProgressBar)
            self.classification_thread.classification_done.connect(self.classificationFinished)
            self.classification_thread.start()
        else:
            QMessageBox.warning(self, "Warning", "Please select both source and destination folders.")

    def updateProgressBar(self, value):
        self.progressBar.setValue(value)

    def classificationFinished(self, elapsed_time_str):
        QMessageBox.information(self, "Classification Completed", f"Images have been classified and organized.\nTime taken: {elapsed_time_str}.")


if __name__ == '__main__':
    # Setup the Model
    model_path = resource_path('models/model_trained.pth')
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)  # Adjust according to your number of classes
    #model.load_state_dict(torch.load('models/model_trained.pth'))
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to inference mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the Transform
    # Adjust the transform to ensure all images are converted to RGB
    transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),  # Convert image to RGB
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class_names = ['Landscapes', 'Memes', 'People', 'Various']
    
    app = QApplication(sys.argv)
    ex = App(model, device, transform, class_names)
    ex.show()
    sys.exit(app.exec_())
