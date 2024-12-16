import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from cnn_model import CNN
from load_emnist import load_emnist_images, load_emnist_labels

# 1. Veri setini yükle
train_images = load_emnist_images('../data/emnist-balanced-train-images-idx3-ubyte.gz')
train_labels = load_emnist_labels('../data/emnist-balanced-train-labels-idx1-ubyte.gz')

# 2. Veri setini PyTorch veri yapısına dönüştür
train_images = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1) / 255.0  # Normalize
train_labels = torch.tensor(train_labels, dtype=torch.long)

# 3. DataLoader oluştur
train_dataset = TensorDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 4. Modeli oluştur ve eğit
model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(2):  # 2 epoch
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
