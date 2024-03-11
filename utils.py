from typing import Tuple
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate the model on the validation set.

    Args:
        model: The model to evaluate.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        device: The device to run the model on.

    Returns:
        A tuple containing the average loss and accuracy on the validation set.
    """
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
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


def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    num_epochs: int = 25,
    save_path: str = "models/model_trained.pth",
) -> nn.Module:
    """
    Train the model and evaluate it on the validation set after each epoch.

    Args:
        model: The model to train.
        criterion: Loss function.
        optimizer: Optimizer for model parameters.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: The device to run the model on.
        writer: TensorBoard SummaryWriter for logging.
        num_epochs: Number of epochs to train for.
        save_path: Path to save the best model.

    Returns:
        The trained model with the best validation loss.
    """
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + i)

        epoch_loss = running_loss / len(train_loader)
        # Calculate train accuracy
        _, train_accuracy = evaluate_model(model, train_loader, criterion, device)
        # Calculate val accuracy
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        writer.add_scalar("Loss/train_avg", epoch_loss, epoch)
        writer.add_scalar("Loss/val_avg", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)
        writer.flush()

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, "
            f"Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

    model.load_state_dict(torch.load(save_path))

    return model


def evaluate_model_per_class(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    test_eval: bool = False,
) -> None:
    """
    Evaluate the model per class on the validation set and print classification report
    and confusion matrix.

    Args:
        model: The model to evaluate.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        device: The device to run the model on.
        test_eval: Flag to indicate if the evaluation is on the test set.
    """
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    classes = (
        val_loader.dataset.classes if test_eval else val_loader.dataset.dataset.classes
    )

    print(classification_report(all_labels, all_preds, target_names=classes))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
