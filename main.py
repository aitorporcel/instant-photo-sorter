import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Transformaciones de los datos
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar las imágenes para el modelo
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Cargar el dataset completo
dataset = ImageFolder(root='dataset_modelo', transform=transform)

# Tamaños para dividir: por ejemplo, 70% entrenamiento, 15% validación, 15% prueba
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Dividir el dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Crear DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Cargar modelo preentrenado
model = models.resnet50(pretrained=True)

# Congelar los parámetros del modelo para no ser entrenados
for param in model.parameters():
    param.requires_grad = False

# Personalizar la última capa para las 4 categorías
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

model = model.to(device)

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

def evaluate_model(model, val_loader, criterion):
    model.eval()  # Poner el modelo en modo de evaluación
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Desactivar el cálculo de gradientes
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Modificar la función de entrenamiento para incluir la evaluación
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()  # Poner el modelo en modo de entrenamiento
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    return model

# Asegúrate de pasar `val_loader` a la función de entrenamiento
model_trained = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10)

# Asumiendo que val_dataset es tu objeto Subset creado a partir de un ImageFolder
# y val_loader se creó a partir de val_dataset

def evaluate_model_per_class(model, val_loader, criterion):
    model.eval()  # Poner el modelo en modo evaluación
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Acceder a las clases desde el dataset original a través del loader
    classes = val_loader.dataset.dataset.classes  # Nota el `.dataset.dataset` para acceder al ImageFolder original

    # Calcular métricas de desempeño por clase
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Opcional: Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    print("Matriz de Confusión:")
    print(cm)

# Después del entrenamiento, evaluar el modelo en el conjunto de validación
evaluate_model_per_class(model_trained, val_loader, criterion)

# Guardar el modelo entrenado
torch.save(model_trained.state_dict(), 'models/model_trained.pth')
