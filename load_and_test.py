import torch

from torch import nn
from networks.FashionMNIST import Linear as NeuralNetwork, test_data, training_data

classes = [
    "T-shirt/Top",
    "Trousers",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()
for i in range(500):
    X, y = test_data[i][0], test_data[i][1]
    print(f"Shape of X [N, C, H, W]: {X.shape}, y = {y}")
    with torch.no_grad():
        X = X.to(device)
        pred = model(X)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicte: "{predicted}", Actual: "{actual}"')
