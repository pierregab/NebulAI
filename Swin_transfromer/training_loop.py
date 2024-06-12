import torch.optim as optim
import torch.nn as nn
import torch
from Swin_transformer_based_model import SwinTransformerObjectDetection, SwinTransformerBackbone, RPN
from dataset import get_data_loaders

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_bbox = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for imgs, bboxes, labels in train_loader:
            imgs = imgs.to(device)
            bboxes = bboxes.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            cls_logits, bbox_regression, mask_pred = model(imgs)
            
            cls_loss = criterion_cls(cls_logits, labels)
            bbox_loss = criterion_bbox(bbox_regression, bboxes)
            loss = cls_loss + bbox_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, bboxes, labels in val_loader:
                imgs = imgs.to(device)
                bboxes = bboxes.to(device)
                labels = labels.to(device)

                cls_logits, bbox_regression, mask_pred = model(imgs)

                cls_loss = criterion_cls(cls_logits, labels)
                bbox_loss = criterion_bbox(bbox_regression, bboxes)
                loss = cls_loss + bbox_loss

                val_loss += loss.item()
        
        print(f'Validation Loss: {val_loss/len(val_loader)}')

    print('Training completed.')

# Initialize the model
img_size = 512
in_chans = 3  # Assuming RGB images
backbone = SwinTransformerBackbone(img_size=img_size, in_chans=in_chans)
rpn = RPN(in_channels=backbone.num_layers * 96)
model = SwinTransformerObjectDetection(backbone, rpn)

# Load the data
annotations_file = 'annotations.json'
train_loader, val_loader = get_data_loaders(annotations_file)

# Train the model
train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001)
