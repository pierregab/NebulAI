import torch
from dataset import get_data_loaders
from Swin_transformer_based_model import SwinTransformerObjectDetection, SwinTransformerBackbone, RPN
from torchvision import transforms
from torch import nn, optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import precision_score, recall_score

def visualize_detections(image, boxes, cls_logits, threshold=0.5):
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)

    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)

    for i in range(len(boxes)):
        score = torch.sigmoid(cls_logits[i])[1]
        if score > threshold:
            box = boxes[i]

            rect = patches.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)

    plt.axis('off')
    plt.show()

def train(model, train_loader, val_loader, device, num_epochs=10, lr=0.001):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_bbox = nn.SmoothL1Loss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for imgs, bboxes, labels in train_loader:
            imgs, bboxes, labels = imgs.to(device), bboxes.to(device), labels.to(device)

            optimizer.zero_grad()
            cls_logits, bbox_regression, _ = model(imgs)

            loss_cls = criterion_cls(cls_logits, labels)
            loss_bbox = criterion_bbox(bbox_regression, bboxes)
            loss = loss_cls + loss_bbox

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

        validate(model, val_loader, device)

def validate(model, val_loader, device, threshold=0.5):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for imgs, bboxes, labels in val_loader:
            imgs, bboxes, labels = imgs.to(device), bboxes.to(device), labels.to(device)
            cls_logits, bbox_regression, _ = model(imgs)

            preds = (torch.sigmoid(cls_logits)[:, 1] > threshold).long()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    print(f"Validation Precision: {precision}, Recall: {recall}")

if __name__ == "__main__":
    annotations_file = 'annotations.json'
    train_loader, val_loader = get_data_loaders(annotations_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create backbone and ensure the correct output channels for the RPN
    backbone = SwinTransformerBackbone()
    rpn_in_channels = backbone.layers[-1][0].block1.dim
    rpn = RPN(in_channels=rpn_in_channels)

    model = SwinTransformerObjectDetection(backbone, rpn)
    
    train(model, train_loader, val_loader, device, num_epochs=10)