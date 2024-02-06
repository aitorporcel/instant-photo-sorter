import torch
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, val_loader, criterion, device):
    """
    Evalua el modelo en el conjunto de validación

    Args:
    model: Modelo a evaluar
    val_loader: Cargador de validación
    
    """
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
def train_model(model, criterion, optimizer, train_loader, val_loader, device, writer,
                num_epochs=25, save_path='models/model_trained.pth'):
    
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()  # Poner el modelo en modo de entrenamiento
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)

        epoch_loss = running_loss / len(train_loader)
        # Calculate train accuracy
        _, train_accuracy = evaluate_model(model, train_loader, criterion, device)
        # Calculate val accuracy
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model
            torch.save(model.state_dict(), save_path)

        # Write the accuracy and loss in tensorboard
        writer.add_scalar('Loss/train_avg', epoch_loss, epoch)
        writer.add_scalar('Loss/val_avg', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        writer.flush()
    

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    # Load the best model
    model.load_state_dict(torch.load(save_path))

    return model

def evaluate_model_per_class(model, val_loader, criterion, device, test_eval=False):
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

    if test_eval:
        classes = val_loader.dataset.classes  # Nota el `.dataset` para acceder al ImageFolder original
    else:
        # Acceder a las clases desde el dataset original a través del loader
        classes = val_loader.dataset.dataset.classes  # Nota el `.dataset.dataset` para acceder al ImageFolder original

    # Calcular métricas de desempeño por clase
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Opcional: Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    print("Matriz de Confusión:")
    print(cm)