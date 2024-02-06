import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import models
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
import datetime
from utils import evaluate_model, train_model, evaluate_model_per_class

# Current date
now = datetime.datetime.now()

# Use the data in the experiment name including the time
experiment_name = f"experiment_{now.strftime('%d_%m_%Y_%H_%M_%S')}"

# Create a SummaryWriter to write logs to
writer = SummaryWriter(f'runs/{experiment_name}')

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Transformaciones de los datos
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Cargar el dataset completo
dataset = ImageFolder(root='dataset_modelo', transform=transform)
test_dataset = ImageFolder(root='test_data', transform=transform_test)


# Tamaños para dividir: por ejemplo, 70% entrenamiento, 15% validación, 15% prueba
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size

# Dividir el dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Less batch size is usually good for regularization
batch_size = 16 #initial 32

# Crear DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Cargar modelo preentrenado
model = models.resnet50(pretrained=True)

# Congelar los parámetros del modelo para no ser entrenados
for param in model.parameters():
    param.requires_grad = False

# Personalizar la última capa para las 4 categorías
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

# Unfreeze the last few layers
for param in model.layer4.parameters():
    param.requires_grad = True

model = model.to(device)

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.fc.parameters(), lr=0.0005) #Initial 0.001

# Define the optimizer with specific parameters and learning rates
optimizer = optim.Adam([
    {'params': model.fc.parameters()},
    {'params': model.layer4.parameters(), 'lr': 1e-5}
], lr=0.0005)


# Asegúrate de pasar `val_loader` a la función de entrenamiento
model_trained = train_model(model, criterion, optimizer, train_loader, val_loader, device, writer, num_epochs=20)

# Asumiendo que val_dataset es tu objeto Subset creado a partir de un ImageFolder
# y val_loader se creó a partir de val_dataset

# Después del entrenamiento, evaluar el modelo en el conjunto de validación
evaluate_model_per_class(model_trained, val_loader, criterion, device, test_eval=False)

# Después del entrenamiento, evaluar el modelo en el conjunto de testeo
evaluate_model_per_class(model_trained, test_loader, criterion, device, test_eval=True)

# Guardar el modelo entrenado
#torch.save(model_trained.state_dict(), 'models/model_trained.pth')
