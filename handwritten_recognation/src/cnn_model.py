import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # (28, 28) -> (28, 28)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # (28, 28) -> (14, 14)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (14, 14) -> (14, 14)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (14, 14) -> (7, 7)

        # 64x7x7 boyutuna göre fully connected layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Bu boyut 64 * 7 * 7
        self.fc2 = nn.Linear(128, 47)  # 47 sınıf (A-Z, a-z, 0-9)

    def forward(self, x):
        print(f"Input Shape: {x.shape}")  # Giriş boyutunu kontrol et

        # İlk convolution + pooling işlemi
        x = self.pool(torch.relu(self.conv1(x)))
        print(f"Shape after conv1 + pool: {x.shape}")

        # İkinci convolution + pooling işlemi
        x = self.pool2(torch.relu(self.conv2(x)))
        print(f"Shape after conv2 + pool2: {x.shape}")

        # Tensörü düzleştir
        x = x.view(x.size(0), -1)  # Otomatik flatten
        print(f"Shape after flatten: {x.shape}")

        # Fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Test verisi oluştur
x = torch.randn(8, 1, 28, 28)  # batch_size=8, channels=1, height=28, width=28
model = CNN()
output = model(x)
print(f"Final output shape: {output.shape}")  # Final shape (batch_size, 47)
