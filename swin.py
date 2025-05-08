import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from timm import create_model
from timm.data import Mixup
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split

# Define Center Loss
class CenterLoss(nn.Module):
    def __init__(self, num_classes=200, feat_dim=1024, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(beta=1, alpha=-2, mat1=x, mat2=self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss

# Define BirdDataset
class BirdDataset(Dataset):
    def __init__(self, df, transform=None):
        self.data = df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        label = int(self.data.iloc[idx]['label']) - 1  # Shift labels from 1–200 to 0–199
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Define TestDataset (no labels)
class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.data = df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

# Create CSV from train folder and split
def prepare_train_data(train_dir):
    data = {'image_path': [], 'label': []}
    for class_folder in sorted(os.listdir(train_dir)):
        class_path = os.path.join(train_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        label = int(class_folder.split('.')[0])  # e.g., "001" -> 1
        for img_name in os.listdir(class_path):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_path, img_name)
                data['image_path'].append(img_path)
                data['label'].append(label)
    
    df = pd.DataFrame(data)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_df.to_csv('train_split.csv', index=False)
    val_df.to_csv('val_split.csv', index=False)
    return train_df, val_df

# Create CSV for test folder
def prepare_test_data(test_dir):
    data = {'image_path': []}
    for img_name in sorted(os.listdir(test_dir)):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(test_dir, img_name)
            data['image_path'].append(img_path)
    test_df = pd.DataFrame(data)
    test_df.to_csv('test.csv', index=False)
    return test_df

# Define transforms with 384x384 input size
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(384),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths
train_dir = 'data/train'  # 200 subfolders
test_dir = 'data/test'    # Flat folder with test images

# Prepare data
train_df, val_df = prepare_train_data(train_dir)
test_df = prepare_test_data(test_dir)

# Initialize datasets and loaders
train_dataset = BirdDataset(train_df, transform=train_transforms)
val_dataset = BirdDataset(val_df, transform=val_transforms)
test_dataset = TestDataset(test_df, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=4)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Swin-B model
model = create_model('swin_base_patch4_window12_384_in22k', pretrained=True, num_classes=200)
model = model.to(device)

# Define Mixup with CutMix
mixup_fn = Mixup(mixup_alpha=0.0, cutmix_alpha=1.0, num_classes=200)

# Define loss functions
ce_loss_fn = nn.CrossEntropyLoss()
center_loss_fn = CenterLoss(num_classes=200, feat_dim=1024, use_gpu=torch.cuda.is_available())

# Define optimizers
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
center_optimizer = optim.AdamW(center_loss_fn.parameters(), lr=0.001)

# Define scheduler
num_epochs = 100
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# Custom forward function to get features for Center Loss
def forward_with_features(model, x):
    features = model.forward_features(x)  # [B, num_patches, embed_dim]
    features = features.mean(dim=1)       # [B, embed_dim], e.g., [B, 1024]
    logits = model.head(features)         # [B, num_classes]
    return logits, features

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Apply CutMix
        images, targets = mixup_fn(images, labels)

        # Forward pass with features
        logits, features = forward_with_features(model, images)

        # Compute losses
        if isinstance(targets, tuple):
            labels_a, labels_b, lam = targets
            ce_loss = lam * ce_loss_fn(logits, labels_a) + (1 - lam) * ce_loss_fn(logits, labels_b)
            cent_loss = center_loss_fn(features, labels_a)  # Use original labels for Center Loss
        else:
            ce_loss = ce_loss_fn(logits, targets)
            cent_loss = center_loss_fn(features, targets)

        total_loss = ce_loss + 0.005 * cent_loss

        # Backward pass
        optimizer.zero_grad()
        center_optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        center_optimizer.step()

        running_loss += total_loss.item()

    scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Use default forward for validation
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), 'swin_b_model.pth')

# Inference on test set
model.eval()
test_predictions = []
test_image_paths = []
with torch.no_grad():
    for images, img_paths in test_loader:
        images = images.to(device)
        outputs = model(images)  # Use default forward for inference
        _, predicted = torch.max(outputs, 1)
        test_predictions.extend(predicted.cpu().numpy() + 1)  # Shift back to 1–200
        test_image_paths.extend(img_paths)

# Save test predictions
test_results = pd.DataFrame({
    'image_path': test_image_paths,
    'predicted_label': test_predictions
})
test_results.to_csv('test_predictions.csv', index=False)
print("Test predictions saved to 'test_predictions.csv'")