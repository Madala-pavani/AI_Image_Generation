import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Define the Generator class (matching trained model)
class Generator(nn.Module):
    def __init__(self, z_size):
        super(Generator, self).__init__()
        self.fc = nn.Linear(z_size, 512 * 8 * 8)

        self.tconv1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)  # 16x16
        self.tconv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)  # 32x32
        self.tconv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)   # 64x64
        self.tconv4 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)     # 128x128

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 8, 8)
        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        x = F.relu(self.tconv3(x))
        return torch.tanh(self.tconv4(x))

# Load trained Generator model
z_size = 100
model = Generator(z_size)
model.load_state_dict(torch.load("generator.pth", map_location=torch.device("cpu")))
model.eval()


# Function to generate an image
def generate_image():
    z = torch.randn(1, z_size)
    with torch.no_grad():
        fake_image = model(z).cpu().numpy()
    return np.transpose(fake_image[0], (1, 2, 0))  # Convert for display