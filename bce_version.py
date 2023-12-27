import torch
from torchvision import datasets, transforms
from torch import nn, optim

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a transform to normalize the data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# Download and load the training data
trainset = datasets.MNIST(
    "~/.pytorch/MNIST_data/", download=True, train=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Define a simple ANN with two hidden layers and one output layer
model = nn.Sequential(
    nn.Linear(784, 128),  # Input layer (784 inputs, 128 outputs)
    nn.ReLU(),  # Activation function (ReLU)
    nn.Linear(128, 64),  # Hidden layer (128 inputs, 64 outputs)
    nn.ReLU(),  # Activation function (ReLU)
    nn.Linear(64, 10),  # Output layer (64 inputs, 10 outputs)
    nn.LogSoftmax(dim=1),
)  # Activation function (LogSoftmax)

# Move the model to the device (GPU if available, else CPU)
model.to(device)

# Define the loss function (Negative Log Likelihood Loss)
criterion = nn.NLLLoss()

# Define the optimizer (Stochastic Gradient Descent) with learning rate 0.003
optimizer = optim.SGD(model.parameters(), lr=0.003)

# Set the number of training epochs
epochs = 5

# Start the training loop
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Move images and labels to the device
        images, labels = images.to(device), labels.to(device)

        # Training pass
        optimizer.zero_grad()  # Clear the gradients

        output = model(images)  # Forward pass
        loss = criterion(output, labels)  # Compute the loss

        # Backpropagation
        loss.backward()  # Compute the gradients

        # Update weights
        optimizer.step()  # Update the weights

        running_loss += loss.item()
    else:
        print(
            f"Training loss: {running_loss/len(trainloader)}"
        )  # Print the training loss

# Download and load the testing data
testset = datasets.MNIST(
    "~/.pytorch/MNIST_data/", download=True, train=False, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Initialize the test loss and the accuracy
test_loss = 0
accuracy = 0

# Switch model to evaluation mode. This means that dropout and batch norm are disabled.
model.eval()

# Turn off gradients for testing
with torch.no_grad():
    for images, labels in testloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Move images and labels to the device
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        log_ps = model(images)

        # Compute the loss
        test_loss += criterion(log_ps, labels)

        # Get the class probabilities
        ps = torch.exp(log_ps)

        # Get the top result
        top_p, top_class = ps.topk(1, dim=1)

        # Check how many of the classes were correct?
        equals = top_class == labels.view(*top_class.shape)

        # Calculate the accuracy
        accuracy += torch.mean(equals.type(torch.FloatTensor))

# Switch model back to train mode
model.train()

# Print the test loss and the accuracy
print(f"Test loss: {test_loss/len(testloader)}")
print(f"Test accuracy: {accuracy/len(testloader)}")
