import torch
from cnn_model import CNN

model_path = 'model_weights.pth'  # Model ağırlıklarının kaydedildiği yol

# Modeli oluştur
model = CNN()

# Cihazı belirle (GPU varsa kullan)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Model ağırlıklarını yükle
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model başarıyla yüklendi.")
except FileNotFoundError:
    print(f"Model dosyası bulunamadı: {model_path}")
except EOFError:
    print(f"Model dosyası boş veya bozuk: {model_path}")
except Exception as e:
    print(f"Model yüklenirken bir hata oluştu: {e}")

# Modeli değerlendirme moduna al
model.eval()

# Örnek bir test işlemi (isteğe bağlı)
# import numpy as np
# from load_emnist import load_emnist_images, load_emnist_labels
# test_images = load_emnist_images('path_to_test_images.gz')
# test_labels = load_emnist_labels('path_to_test_labels.gz')
# test_images = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1) / 255.0
# test_labels = torch.tensor(test_labels, dtype=torch.long)
# test_loader = DataLoader(TensorDataset(test_images, test_labels), batch_size=64, shuffle=False)

# with torch.no_grad():
#     total = 0
#     correct = 0
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     print(f"Doğruluk: {100 * correct / total}%")
