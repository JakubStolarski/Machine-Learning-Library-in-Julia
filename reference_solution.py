import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as f
import torch.optim as optim


# creating a network model based on the topology of LeNet
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))
        x = f.max_pool2d(f.relu(self.conv2(x)), (2, 2))
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def training(device, num_of_epochs, model, train_loader, criterion, optimizer):
    num_of_step = len(train_loader)
    for epoch in range(num_of_epochs):
        for i, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)

            outputs = model(image)
            loss = criterion(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 400 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_of_epochs, i + 1, num_of_step, loss.item()))


def testing(test_loader, model):
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Network().to(device)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.016
    num_of_epochs = 10
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    # Loading the dataset and preprocessing
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.Compose([
                                                   transforms.Resize((32, 32)),
                                                   transforms.ToTensor()]),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.Compose([
                                                  transforms.Resize((32, 32)),
                                                  transforms.ToTensor()]),
                                              download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)
    dataset_sizes = {'train': len(train_dataset), 'test': len(test_dataset)}

    training(device, num_of_epochs, net, train_loader, criterion, optimizer)
    testing(test_loader, net)


