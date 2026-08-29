"""A module containing functions for training and testing a PyTorch neural network on the MNIST dataset,
as well as data loaders for the dataset. Running will download the required MNIST dataset, if not present."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import joblib
from torch.utils.data import DataLoader
from time import perf_counter
from file_name_funcs import get_valid_file_name

def load_model(file_name = "number_identifier_models\\cnn_number_identifier_v3.joblib"):
    """Loads and returns and image classifier model. Make sure to also import Net in order to ensure functionality."""
    cnn = joblib.load(file_name)
    return cnn

def get_mnist_loaders(batch_size: int = 64, shuffle_train: bool = True) -> tuple:
    """Get MNIST data loaders for training and testing.

    Parameters:
        batch_size: int, optional, batch size for data loaders, default = 64
        shuffle_train: bool, optional, shuffle training data, default = True

    Returns:
        Tuple of train_loader and test_loader data loaders.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    augment = transforms.Compose([
        transforms.RandomAffine(
            degrees=12,
            translate=(0.20, 0.20),
            scale=(0.40, 1.10),
            shear=(-5, 5),
            fill=0,
        ),
        transforms.RandomApply([
            transforms.ElasticTransform(
                alpha=5.0,
                sigma=4.0,
            )
        ], p=0.15),
        transforms.RandomPerspective(
            distortion_scale=0.10,
            p=0.20,
            fill=0,
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.20,
            scale=(0.01, 0.08),
            ratio=(0.2, 5.0),
            value=0,
        ),
        transforms.RandomApply([transforms.Lambda(lambda x: (x > 0.5).float())], p=0.5),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_dataset = torchvision.datasets.MNIST(root='./datasets', train=True, transform=augment, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./datasets', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train_and_test(model: torch.nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                   device: str, num_epochs: int = 5):
    """Train and test the model.

    Parameters:
        model: torch.nn.Module, the neural network model
        train_loader: DataLoader, data loader for training data
        test_loader: DataLoader, data loader for test data
        device: str, 'cpu', 'cuda', or 'mps' device for training
        num_epochs: int, optional, number of training epochs, default = 5
    """

    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("Training model...")
    duration = 0
    eval_history = list()
    early_stopping = 5                      # Stop after 2 epochs without improvement
    minimum_improvement = 1.01              # Early stopping threshold
    for epoch in range(num_epochs):
        start_time = perf_counter()
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print("Testing model...")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        duration += perf_counter() - start_time
        accuracy = 100 * correct / total

        print(f"Epoch {epoch+1}, Test Accuracy: {accuracy:.2f}%\t\tDuration: {round(duration, 2)} seconds.")

        eval_history.append(accuracy)
        recent_accuracy = sum(eval_history[-early_stopping-1:-1]) / early_stopping

        if len(eval_history) > 3 and accuracy < recent_accuracy * minimum_improvement:
            print("Early stopping.")
            break

    if input("Save model? ").upper() in ("Y", "YES"):
        path = "number_identifier_models"
        file_name = get_valid_file_name(path, "cnn_number_identifier_v", file_extension= ".joblib")
        joblib.dump(model, file_name)
        print(f"Saved as {file_name}")

def main():
    # Get cpu, gpu, or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu")
    print(f"\nUsing {device} device.\n")

    train_loader, test_loader = get_mnist_loaders(batch_size=128)

    nn = Net()
    num_epochs = 10

    train_and_test(nn, train_loader, test_loader, device, num_epochs)

if __name__ == "__main__":
    main()
