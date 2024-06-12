import torch
from dataset import get_data_loaders
from Swin_transformer_based_model import SwinTransformerObjectDetection, SwinTransformerBackbone, RPN
from torchvision import transforms
from torch import nn, optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import precision_score, recall_score

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
    yB = min(boxB[3], boxA[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
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

def train(model, train_loader, val_loader, device, num_epochs=10, lr=0.001):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_bbox = nn.SmoothL1Loss()

    model.set_debug(False)  # Disable debugging

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        print(f"Starting epoch {epoch+1}/{num_epochs}")

        for i, (imgs, bboxes, labels) in enumerate(train_loader):
            print(f"Processing batch {i+1}/{len(train_loader)}")
            imgs, bboxes, labels = imgs.to(device), bboxes.to(device), labels.to(device)

            optimizer.zero_grad()
            cls_logits, bbox_regression, _ = model(imgs)

            # Match proposals with ground truth
            proposals = bbox_regression.view(-1, 4)
            matched_proposals, matched_ground_truths = match_proposals_with_ground_truth(proposals, bboxes.view(-1, 4))

            if not matched_proposals:
                continue

            matched_proposals = torch.stack(matched_proposals)
            matched_ground_truths = torch.stack(matched_ground_truths)

            loss_cls = criterion_cls(cls_logits, labels)
            loss_bbox = criterion_bbox(matched_proposals, matched_ground_truths)
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
        for i, (imgs, bboxes, labels) in enumerate(val_loader):
            print(f"Validating batch {i+1}/{len(val_loader)}")
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
    backbone = SwinTransformerBackbone(debug=False)
    rpn_in_channels = backbone.layers[-1][0].block1.dim
    rpn = RPN(in_channels=rpn_in_channels, debug=False)

    model = SwinTransformerObjectDetection(backbone, rpn, debug=False)
    
    print("Starting training process")
    train(model, train_loader, val_loader, device, num_epochs=10)
    print("Training process finished")
