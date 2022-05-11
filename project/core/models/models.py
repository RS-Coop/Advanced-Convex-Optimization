'''
Various models used in SARC experiments
'''
import torch
import torch.nn.functional as F

'''
Simple CNN for image classification with 2 convolutional layers
'''
class CNN2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 16, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, 5)
        self.fc1 = torch.nn.Linear(32*5*5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)

    def predict(self, x):
        out = self.forward(x)
        return torch.argmax(F.softmax(out, dim=1), dim=1)

'''
Simple dense network for image classification with 3 fully connected layers
'''
class Dense3(torch.nn.Module):
    def __init__(self, n1=128, n2=128):
        super().__init__()

        self.fc1 = torch.nn.Linear(28*28, n1)
        self.fc2 = torch.nn.Linear(n1, n2)
        self.fc3 = torch.nn.Linear(n2, 10)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        return self.fc3(x)

    def predict(self, x):
        out = self.forward(x)
        return torch.argmax(F.softmax(out, dim=1), dim=1)

'''
Simple 1 layer dense network for logistic regression
'''
class Dense1(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.linear = torch.nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        outputs = self.forward(x)
        return torch.round(torch.sigmoid(outputs))

'''
Simple 2 layer dense network for logistic regression
'''
class Dense2(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.fc1 = torch.nn.Linear(input_dim, 10, bias=False)
        self.fc2 = torch.nn.Linear(10, 1, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x))

    def predict(self, x):
        outputs = self.forward(x)
        return torch.round(torch.sigmoid(outputs))
