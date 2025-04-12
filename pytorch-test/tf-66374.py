import torch
 import torch.nn as nn
 import torch.optim as optim
 import torch.utils.data as data
 from torchvision import datasets, transforms

 # Set global policy for mixed precision (if needed) - no direct equivalent in PyTorch,
 # consider using torch.cuda.amp if available and desired
 # keras.mixed_precision.set_global_policy("mixed_float16")

 # Define the input layer
 input_size = 784
 # Define the dense layer
 class SimpleModel(nn.Module):
  def __init__(self):
   super(SimpleModel, self).__init__()
   self.fc = nn.Linear(input_size, 10) # Fully connected layer
   self.softmax = nn.Softmax(dim=1) # Softmax activation

  def forward(self, x):
   x = self.fc(x)
   x = self.softmax(x) # Apply softmax
   return x

 model = SimpleModel()

 # Define the loss function
 loss_fn = nn.CrossEntropyLoss()

 # Define the optimizer
 optimizer = optim.RMSprop(model.parameters())

 # Load the MNIST dataset
 train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
 train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)

 # Reshape and normalize the training data (done via transforms in PyTorch)
 # x_train = x_train.reshape(60000, 784).astype("float32") / 255 # Handled by transform

 # Training loop
 num_epochs = 1
 for epoch in range(num_epochs):
  for batch_idx, (data, target) in enumerate(train_loader):
   # data = data.reshape(-1, 784) # Reshape is not needed because of DataLoader
   optimizer.zero_grad()
   output = model(data)
   loss = loss_fn(output, target)
   loss.backward()
   optimizer.step()

  # verbose = 0, so no print statements
  # _ = model.fit(x_train, y_train, batch_size=128, epochs=1, steps_per_epoch=1, verbose=0)