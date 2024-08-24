import torch
from dataset import get_data_loaders
from Swin_transformer_based_model import SwinTransformerObjectDetection, SwinTransformerBackbone, RPN
from torchvision import transforms
from torch import nn, optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import subprocess
import torchvision
import os

def match_proposals_with_ground_truth(proposals, ground_truths, iou_threshold=0.5):
    matched_proposals = []
    matched_ground_truths = []
    for gt in ground_truths:
        max_iou = 0
        best_proposal = None
        for proposal in proposals:
            iou = calculate_iou(gt, proposal)
            if iou > max_iou:
                max_iou = iou
                best_proposal = proposal
        if max_iou >= iou_threshold:
            matched_proposals.append(best_proposal)
            matched_ground_truths.append(gt)
    return matched_proposals, matched_ground_truths

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

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

def start_tensorboard(log_dir): 
    try:
        os.makedirs(log_dir, exist_ok=True)
        process = subprocess.Popen(['tensorboard', '--logdir', log_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(f"TensorBoard started at http://localhost:6006")
        if stdout:
            print(f"TensorBoard stdout: {stdout.decode('utf-8')}")
        if stderr:
            print(f"TensorBoard stderr: {stderr.decode('utf-8')}")
    except Exception as e:
        print(f"Failed to start TensorBoard: {e}")

def train(model, train_loader, val_loader, device, num_epochs=10, lr=0.001, log_dir="runs/debug"):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_bbox = nn.SmoothL1Loss()
    
    writer = SummaryWriter(log_dir=log_dir)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        print(f"Starting epoch {epoch+1}/{num_epochs}")

        for i, (imgs, bboxes, labels) in enumerate(train_loader):
            print(f"Processing batch {i+1}/{len(train_loader)}")
            imgs, bboxes, labels = imgs.to(device), bboxes.to(device), labels.to(device)

            optimizer.zero_grad()
            cls_logits, bbox_regression, _ = model(imgs)

            proposals = bbox_regression.view(-1, 4)
            matched_proposals, matched_ground_truths = match_proposals_with_ground_truth(proposals, bboxes.view(-1, 4))

            if not matched_proposals:
                continue

            matched_proposals = torch.stack(matched_proposals).to(device)
            matched_ground_truths = torch.stack(matched_ground_truths).to(device)

            loss_cls = criterion_cls(cls_logits, labels)
            loss_bbox = criterion_bbox(matched_proposals, matched_ground_truths)
            loss = loss_cls + loss_bbox

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            writer.add_scalar('Loss/Total', running_loss/(i+1), epoch*len(train_loader)+i)
            writer.add_scalar('Loss/Classification', loss_cls.item(), epoch*len(train_loader)+i)
            writer.add_scalar('Loss/Regression', loss_bbox.item(), epoch*len(train_loader)+i)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

        precision, recall = validate(model, val_loader, device, writer, epoch)
        writer.add_scalar('Precision', precision, epoch)
        writer.add_scalar('Recall', recall, epoch)

    writer.close()

def validate(model, val_loader, device, writer, epoch, threshold=0.5):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for i, (imgs, bboxes, labels) in enumerate(val_loader):
            print(f"Validating batch {i+1}/{len(val_loader)}")
            imgs, bboxes, labels = imgs.to(device), bboxes.to(device), labels.to(device)
            cls_logits, bbox_regression, _ = model(imgs)

            # If cls_logits has shape [batch_size * num_proposals, num_classes] 
            # Here batch_size*num_proposals=62 and num_classes=2

            # Assuming you want the scores for the second class (index 1)
            scores = torch.sigmoid(cls_logits[:, 1])  # Shape: (62,)

            # Predict positive if the score is above the threshold
            batch_preds = (scores > threshold).long()  # Shape: (62,)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(batch_preds.cpu().numpy())

            if i == 0:
                img_grid = torchvision.utils.make_grid(imgs)
                writer.add_image('Validation Images', img_grid, epoch)

                if bbox_regression.numel() > 0:  # Check if bbox_regression is not empty
                    visualize_detections(imgs[0], bbox_regression[0], cls_logits[0], threshold)

    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')

    print(f"Validation Precision: {precision}, Recall: {recall}")

    return precision, recall

if __name__ == "__main__":
    annotations_file = 'annotations.json'
    train_loader, val_loader = get_data_loaders(annotations_file, batch_size=1, num_images=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    backbone = SwinTransformerBackbone(img_size=256)
    rpn_in_channels = backbone.layers[-1][0].block1.dim
    rpn = RPN(in_channels=rpn_in_channels)
    
    model = SwinTransformerObjectDetection(backbone, rpn)
    
    print("Starting training process")
    train(model, train_loader, val_loader, device, num_epochs=10)
    print("Training process finished")
