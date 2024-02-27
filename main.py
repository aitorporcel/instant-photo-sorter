import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

from utils import train_model, evaluate_model_per_class

def main(num_epochs: int, batch_size: int, train_split: float):
    now = datetime.datetime.now()
    experiment_name = f"experiment_{now.strftime('%d_%m_%Y_%H_%M_%S')}"
    writer = SummaryWriter(f'runs/{experiment_name}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),  # Convert image to RGB
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),  # Convert image to RGB
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolder(root='train_data', transform=transform)
    test_dataset = ImageFolder(root='test_data', transform=transform_test)

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    for param in model.layer4.parameters():
        param.requires_grad = True

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.fc.parameters()},
        {'params': model.layer4.parameters(), 'lr': 1e-5}
    ], lr=0.0005)

    model_trained = train_model(model, criterion, optimizer, train_loader, val_loader,
                                device, writer, num_epochs=num_epochs)
    
    evaluate_model_per_class(model_trained, val_loader, criterion, device, test_eval=False)
    evaluate_model_per_class(model_trained, test_loader, criterion, device, test_eval=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on a dataset.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and validation.")
    parser.add_argument("--train_split", type=float, default=0.7, help="Percentage of data to use for training (rest is for validation).")

    args = parser.parse_args()

    main(num_epochs=args.num_epochs, batch_size=args.batch_size, train_split=args.train_split)
