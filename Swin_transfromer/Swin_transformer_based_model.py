import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align, nms

class RelativePositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(RelativePositionEncoding, self).__init__()
        self.encoding = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        pos = self.encoding[:seq_len, :]
        return x + pos.unsqueeze(0)

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size=0):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        print(f'SwinTransformerBlock input: {x.shape}')
        shortcut = x
        x = self.norm1(x)
        
        B, L, C = x.shape
        H, W = int(L ** 0.5), int(L ** 0.5)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        
        x = x.view(B, L, C)

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (C ** 0.5)
        attention_map = self.softmax(attention_scores)
        attention_output = torch.matmul(attention_map, V)

        x = shortcut + attention_output
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        
        x = x.view(B, L, C)

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        print(f'SwinTransformerBlock output: {x.shape}')
        return x

class ConsecutiveSwinTransformerBlocks(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super(ConsecutiveSwinTransformerBlocks, self).__init__()
        self.block1 = SwinTransformerBlock(dim, num_heads, window_size, shift_size=0)
        self.block2 = SwinTransformerBlock(dim, num_heads, window_size, shift_size=window_size // 2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

class PatchPartition(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96):
        super(PatchPartition, self).__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        print(f'PatchPartition input: {x.shape}')
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, L, C
        print(f'PatchPartition output: {x.shape}')
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super(PatchMerging, self).__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        print(f'PatchMerging input: {x.shape}')
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.reduction(x)
        print(f'PatchMerging output: {x.shape}')
        return x

class SwinTransformerBackbone(nn.Module):
    def __init__(self, img_size=512, patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7):
        super(SwinTransformerBackbone, self).__init__()
        self.patch_partition = PatchPartition(patch_size, in_chans, embed_dim)
        self.num_layers = len(depths)
        self.layers = nn.ModuleList()
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList()
            for i_block in range(depths[i_layer]):
                layer.append(ConsecutiveSwinTransformerBlocks(embed_dim * 2**i_layer, num_heads[i_layer], window_size))
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.layers.append(PatchMerging(embed_dim * 2**i_layer))

    def forward(self, x):
        print(f'Backbone input: {x.shape}')
        x = self.patch_partition(x)
        B, L, C = x.shape
        H, W = self.patches_resolution
        assert L == H * W, "Input feature has wrong size"

        feature_pyramids = []
        for layer in self.layers:
            if isinstance(layer, PatchMerging):
                x = layer(x, H, W)
                H, W = H // 2, W // 2
            else:
                for block in layer:
                    x = block(x)
            feature_pyramids.append(x.view(B, H, W, -1).permute(0, 3, 1, 2))  # B, C, H, W
            print(f'Feature pyramid shape: {feature_pyramids[-1].shape}')
        return feature_pyramids

class RPN(nn.Module):
    def __init__(self, in_channels, mid_channels=256, n_anchors=9):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(mid_channels, n_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(mid_channels, n_anchors * 4, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        print(f'RPN input: {x.shape}')
        x = self.relu(self.conv(x))
        logits = self.cls_logits(x)
        bbox_pred = self.bbox_pred(x)
        print(f'RPN output logits: {logits.shape}, bbox_pred: {bbox_pred.shape}')
        return logits, bbox_pred

def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=[8, 16, 32]):
    anchors = []
    for scale in scales:
        for ratio in ratios:
            w = base_size * scale * (ratio ** 0.5)
            h = base_size * scale / (ratio ** 0.5)
            anchors.append([-w / 2, -h / 2, w / 2, h / 2])
    return torch.tensor(anchors)

def generate_anchor_boxes(feature_map_size, base_size=16, n_anchors=9):
    anchors = generate_anchors(base_size)
    grid_x = torch.arange(feature_map_size[1]) * base_size
    grid_y = torch.arange(feature_map_size[0]) * base_size
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid = torch.stack([grid_x, grid_y, grid_x, grid_y], dim=-1)
    anchors = anchors.view(1, 1, -1, 4) + grid.view(feature_map_size[0], feature_map_size[1], 1, 4)
    anchors = anchors.view(-1, 4)
    # Duplicate anchors to match the number of predictions
    anchors = anchors.repeat(2, 1)
    return anchors

def generate_proposals(rpn_logits, rpn_bbox_pred, anchor_boxes, image_size, batch_size, nms_thresh=0.7, pre_nms_top_n=6000, post_nms_top_n=300):
    print(f"RPN logits shape: {rpn_logits.shape}")
    print(f"RPN bbox_pred shape: {rpn_bbox_pred.shape}")
    print(f"Anchor boxes shape: {anchor_boxes.shape}")
    
    logits = rpn_logits.permute(0, 2, 3, 1).contiguous().view(-1, 1)
    bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous().view(-1, 4)
    anchors = anchor_boxes.view(-1, 4)

    print(f"Logits shape: {logits.shape}")
    print(f"Bbox_pred shape: {bbox_pred.shape}")
    print(f"Anchors shape: {anchors.shape}")
    
    scores = logits.squeeze()
    print(f"Scores shape: {scores.shape}")
    proposals = bbox_pred + anchors[:bbox_pred.shape[0], :]
    print(f"Proposals shape: {proposals.shape}")

    proposals[:, [0, 2]] = proposals[:, [0, 2]].clamp(0, image_size[1])
    proposals[:, [1, 3]] = proposals[:, [1, 3]].clamp(0, image_size[0])

    keep = nms(proposals, scores, nms_thresh)
    print(f"Number of proposals after NMS: {len(keep)}")

    # Add batch index to the proposals
    keep = keep[:post_nms_top_n]
    batch_indices = torch.arange(batch_size, dtype=proposals.dtype, device=proposals.device).repeat_interleave(len(keep) // batch_size)
    rois = torch.cat([batch_indices[:, None], proposals[keep]], dim=1)

    return rois


class SwinTransformerObjectDetection(nn.Module):
    def __init__(self, backbone, rpn, num_classes=2):
        super(SwinTransformerObjectDetection, self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.num_classes = num_classes
        
        self.cls_head = nn.Sequential(
            nn.Linear(768 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )
        
        self.bbox_head = nn.Sequential(
            nn.Linear(768 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes * 4)
        )
        
        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1)
        )

    def forward(self, x):
        print(f'Model input: {x.shape}')
        feature_pyramids = self.backbone(x)
        rpn_logits, rpn_bbox_pred = [], []
        feature_map = feature_pyramids[-1]
        print(f'RPN input: {feature_map.shape}')
        logits, bbox_pred = self.rpn(feature_map)
        rpn_logits.append(logits)
        rpn_bbox_pred.append(bbox_pred)
        
        feature_map_size = feature_map.size()[2:]
        anchor_boxes = generate_anchor_boxes(feature_map_size, n_anchors=logits.shape[1])

        rois = generate_proposals(rpn_logits[0], rpn_bbox_pred[0], anchor_boxes, x.size()[2:], x.size(0))
        
        pooled_features = roi_align(feature_pyramids[-1], rois, output_size=(7, 7), spatial_scale=1.0 / 16.0)
        print(f'Pooled features: {pooled_features.shape}')
        
        pooled_features_flat = pooled_features.view(pooled_features.size(0), -1)
        cls_logits = self.cls_head(pooled_features_flat)
        bbox_regression = self.bbox_head(pooled_features_flat)

        mask_pred = self.mask_head(pooled_features)

        print(f'Output - cls_logits: {cls_logits.shape}, bbox_regression: {bbox_regression.shape}, mask_pred: {mask_pred.shape}')
        return cls_logits, bbox_regression, mask_pred

# Create backbone and ensure the correct output channels for the RPN
backbone = SwinTransformerBackbone()
# We should determine the correct output channels dynamically
rpn_in_channels = backbone.layers[-1][0].block1.dim
rpn = RPN(in_channels=rpn_in_channels)

model = SwinTransformerObjectDetection(backbone, rpn)

# Dummy input for testing
dummy_input = torch.randn(2, 3, 512, 512)
model(dummy_input)

