"""
Code belongs to Pierre Gabriel Bibal Sobeaux, 2024 

          ⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
  ⣰⣿⣿⣿⣿⣿⣿⢻⣿⣿⠏⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀
⢀⣿⣿⣿⣿⣿⡟⣿⡟⠘⣿⣿⡇⢞⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀
⢸⣿⣿⣿⣿⣿⣷⣿⡇⠀⠹⣿⣧⠼⢾⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⠀⠀⠀⠀⠀
⣿⣿⣿⣿⣿⣿⣿⣹⣗⠒⠋⠙⢿⣆⠈⢫⣿⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡀⠀⠀⠀⠀
⣿⣿⣿⣿⣿⣻⣿⠘⢿⠀⠠⣖⣿⣿⣿⣿⣿⣿⢿⣿⣿⣿⣿⣿⣿⠚⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀
⢹⣿⣿⣿⣿⣿⣿⡄⠀⠣⠀⠻⠋⠀⠘⣿⠿⠛⢆⢻⣿⣿⣿⣿⣿⢈⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀
⣼⣿⣿⣿⣿⠛⣿⡷⣀⠀⠀⠀⠀⠉⠉⠀⠀⠀⠘⣄⣿⣿⣿⣿⣿⣍⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⡀⠀⠀
⡇⣿⣿⣿⣿⡗⠈⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⣿⣿⣿⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⡇⠀⠀
⡇⢿⣿⣿⣿⣷⠾⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⡇⠀⠀
⡇⢸⣿⣿⣿⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣹⡿⢿⠿⠒⠓⠒⠂
⢣⠸⣿⣿⣿⣿⣷⡀⠀⠀⣀⣤⡀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢇⡿⠋⠀⠁⠀⠀⠀⠀⠀
⠸⠀⣿⣿⣿⣿⣿⣷⣄⡀⠘⠭⠄⠀⠀⠀⠀⠀⠀⡗⣿⣿⣿⣿⢸⣿⣿⣿⣿⣿⡟⣽⢿⠞⢀⣀⣀⣀⠀⠀⢀⠀⠒
⠀⢇⢹⣿⢹⣿⣿⣿⣿⠙⣦⡀⠀⠀⠀⠀⣀⡤⠴⢿⣿⣿⣿⡏⠀⣿⡟⣿⡿⠋⢰⠋⡸⠉⠁⠀⠀⢀⡤⠚⠁⠀⠀
⠀⠈⠛⣿⢀⢿⣿⡿⣿⠀⣿⡙⠲⠒⠒⠉⠙⠢⠀⣿⣿⣿⣿⡇⠀⣹⠙⠏⠁⢀⠎⢠⠃⠀⠀⡠⠖⠁⠀⠀⠀⠀⠀
⠀⠀⠀⠹⣿⠀⢻⣷⡿⡐⠋⠁⠀⠀⠀⠀⠀⠀⣰⢟⣿⢏⡞⢻⢷⠏⠀⠀⠀⢸⣰⠃⢀⡤⠊⠀⡠⠴⠒⠉⠀⠀⠀
⠀⠀⠀⠀⠹⣇⠀⠙⠢⠀⠀⠀⠀⠀⠀⠀⠀⠚⠁⠞⢁⣾⠃⡇⢸⣆⠀⠀⠀⢀⣇⡰⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⢻⠀⠀⢸⡿⣧⡀⠀⡼⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⢇⡏⠠⠆⠸⢡⢱⠈⡹⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀


                                                                ___   ___
    /|    / / //   / /  //   ) )  //   / / / /        // | |       / /   
   //|   / / //____    //___/ /  //   / / / /        //__| |      / /    
  // |  / / / ____    / __  (   //   / / / /        / ___  |     / /     
 //  | / / //        //    ) ) //   / / / /        //    | |    / /      
//   |/ / //____/ / //____/ / ((___/ / / /____/ / //     | | __/ /___  

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align, nms
import torchvision.ops as ops
from torchsummary import summary


class RelativePositionEncoding(nn.Module):
    """
    Adds relative position encoding to the input tensor. This helps the model to capture positional information, 
    which is crucial for understanding spatial relationships in the input data.

    Args:
        d_model (int): Dimension of the model.
        max_len (int, optional): Maximum length of the sequence. Default is 512.

    Inputs:
        x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

    Outputs:
        Tensor: Output tensor with added position encoding of shape (batch_size, seq_len, d_model).
    """
    def __init__(self, d_model, max_len=512):
        super(RelativePositionEncoding, self).__init__()
        self.encoding = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        pos = self.encoding[:seq_len, :]
        return x + pos.unsqueeze(0)


class SwinTransformerBlock(nn.Module):
    """
    A single block of the Swin Transformer, including multi-head self-attention with optional shifting for windows.

    Args:
        dim (int): Number of input dimensions.
        num_heads (int): Number of attention heads.
        window_size (int): Size of the attention window.
        shift_size (int, optional): Shift size for the attention window. Default is 0.
        debug (bool, optional): If True, prints debug information. Default is False.

    Inputs:
        x (Tensor): Input tensor of shape (batch_size, seq_len, dim).

    Outputs:
        Tensor: Output tensor of shape (batch_size, seq_len, dim).
    """
    def __init__(self, dim, num_heads, window_size, shift_size=0, debug=False):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.debug = debug

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
        if self.debug:
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
        if self.debug:
            print(f'SwinTransformerBlock output: {x.shape}')
        return x


class ConsecutiveSwinTransformerBlocks(nn.Module):
    """
    A module containing two consecutive Swin Transformer blocks with alternating shift sizes.

    Args:
        dim (int): Number of input dimensions.
        num_heads (int): Number of attention heads.
        window_size (int): Size of the attention window.
        debug (bool, optional): If True, prints debug information. Default is False.

    Inputs:
        x (Tensor): Input tensor of shape (batch_size, seq_len, dim).

    Outputs:
        Tensor: Output tensor after two transformer blocks of shape (batch_size, seq_len, dim).
    """
    def __init__(self, dim, num_heads, window_size, debug=False):
        super(ConsecutiveSwinTransformerBlocks, self).__init__()
        self.block1 = SwinTransformerBlock(dim, num_heads, window_size, shift_size=0, debug=debug)
        self.block2 = SwinTransformerBlock(dim, num_heads, window_size, shift_size=window_size // 2, debug=debug)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class PatchPartition(nn.Module):
    """
    Divides an image into non-overlapping patches and embeds them into a higher-dimensional space.

    Args:
        patch_size (int, optional): Size of each patch. Default is 4.
        in_chans (int, optional): Number of input channels. Default is 3.
        embed_dim (int, optional): Embedding dimension for each patch. Default is 96.
        debug (bool, optional): If True, prints debug information. Default is False.

    Inputs:
        x (Tensor): Input image tensor of shape (batch_size, in_chans, height, width).

    Outputs:
        Tensor: Flattened and embedded patch tensor of shape (batch_size, num_patches, embed_dim).
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, debug=False):
        super(PatchPartition, self).__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.debug = debug

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        if self.debug:
            print(f'PatchPartition input: {x.shape}')
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, L, C
        if self.debug:
            print(f'PatchPartition output: {x.shape}')
        return x


class PatchMerging(nn.Module):
    """
    Merges patches to reduce the spatial dimension and increase the channel dimension.

    Args:
        dim (int): Dimension of the input patches.
        debug (bool, optional): If True, prints debug information. Default is False.

    Inputs:
        x (Tensor): Input tensor of shape (batch_size, num_patches, dim).
        H (int): Height of the patch grid.
        W (int): Width of the patch grid.

    Outputs:
        Tensor: Merged patch tensor of shape (batch_size, new_num_patches, 2 * dim).
    """
    def __init__(self, dim, debug=False):
        super(PatchMerging, self).__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.debug = debug

    def forward(self, x, H, W):
        if self.debug:
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
        if self.debug:
            print(f'PatchMerging output: {x.shape}')
        return x


class SwinTransformerBackbone(nn.Module):
    """
    The Swin Transformer backbone network that consists of patch partitioning, several stages of 
    Swin Transformer blocks, and patch merging layers.

    Args:
        img_size (int, optional): Size of the input image. Default is 512.
        patch_size (int, optional): Size of each patch. Default is 4.
        in_chans (int, optional): Number of input channels. Default is 3.
        embed_dim (int, optional): Embedding dimension for each patch. Default is 96.
        depths (list, optional): Number of Swin Transformer blocks at each stage. Default is [2, 2, 6, 2].
        num_heads (list, optional): Number of attention heads at each stage. Default is [3, 6, 12, 24].
        window_size (int, optional): Size of the attention window. Default is 7.
        debug (bool, optional): If True, prints debug information. Default is False.

    Inputs:
        x (Tensor): Input image tensor of shape (batch_size, in_chans, height, width).

    Outputs:
        list: List of feature pyramid tensors from each stage.
    """
    def __init__(self, img_size=512, patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, debug=False):
        super(SwinTransformerBackbone, self).__init__()
        self.patch_partition = PatchPartition(patch_size, in_chans, embed_dim, debug=debug)
        self.num_layers = len(depths)
        self.layers = nn.ModuleList()
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        self.debug = debug
        
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList()
            for i_block in range(depths[i_layer]):
                layer.append(ConsecutiveSwinTransformerBlocks(embed_dim * 2**i_layer, num_heads[i_layer], window_size, debug=debug))
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.layers.append(PatchMerging(embed_dim * 2**i_layer, debug=debug))

    def forward(self, x):
        if self.debug:
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
            if self.debug:
                print(f'Feature pyramid shape: {feature_pyramids[-1].shape}')
        return feature_pyramids


class RPN(nn.Module):
    """
    Region Proposal Network (RPN) for generating object proposals.

    Args:
        in_channels (int): Number of input channels.
        mid_channels (int, optional): Number of intermediate channels. Default is 256.
        n_anchors (int, optional): Number of anchor boxes. Default is 9.
        debug (bool, optional): If True, prints debug information. Default is False.

    Inputs:
        x (Tensor): Input feature map of shape (batch_size, in_channels, height, width).

    Outputs:
        Tuple[Tensor, Tensor]: 
            logits: Classification scores for each anchor box of shape (batch_size, n_anchors, height, width).
            bbox_pred: Bounding box regression predictions of shape (batch_size, n_anchors * 4, height, width).
    """
    def __init__(self, in_channels, mid_channels=256, n_anchors=9, debug=False):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(mid_channels, n_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(mid_channels, n_anchors * 4, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.debug = debug

    def forward(self, x):
        if self.debug:
            print(f'RPN input: {x.shape}')
        x = self.relu(self.conv(x))
        logits = self.cls_logits(x)
        bbox_pred = self.bbox_pred(x)
        if self.debug:
            print(f'RPN output logits: {logits.shape}, bbox_pred: {bbox_pred.shape}')
        return logits, bbox_pred


def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=[8, 16, 32]):
    """
    Generates anchor boxes based on base size, aspect ratios, and scales.

    Args:
        base_size (int, optional): Base size of the anchor boxes. Default is 16.
        ratios (list, optional): Aspect ratios for the anchor boxes. Default is [0.5, 1, 2].
        scales (list, optional): Scales for the anchor boxes. Default is [8, 16, 32].

    Returns:
        Tensor: Generated anchor boxes of shape (num_anchors, 4).
    """
    anchors = []
    for scale in scales:
        for ratio in ratios:
            w = base_size * scale * (ratio ** 0.5)
            h = base_size * scale / (ratio ** 0.5)
            anchors.append([-w / 2, -h / 2, w / 2, h / 2])
    return torch.tensor(anchors)


def generate_anchor_boxes(feature_map_size, base_size=16, n_anchors=9):
    """
    Generates anchor boxes for the entire feature map based on the specified size.

    Args:
        feature_map_size (tuple): Size of the feature map (height, width).
        base_size (int, optional): Base size of the anchor boxes. Default is 16.
        n_anchors (int, optional): Number of anchor boxes per location. Default is 9.

    Returns:
        Tensor: Generated anchor boxes for the entire feature map of shape (total_anchors, 4).
    """
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


def generate_proposals(rpn_logits, rpn_bbox_pred, anchor_boxes, image_size, batch_size, nms_thresh=0.7, pre_nms_top_n=6000, post_nms_top_n=300, debug=False):
    """
    Generates proposals from the RPN logits and bounding box predictions.

    Args:
        rpn_logits (Tensor): RPN classification logits of shape (batch_size, n_anchors, height, width).
        rpn_bbox_pred (Tensor): RPN bounding box regression predictions of shape (batch_size, n_anchors * 4, height, width).
        anchor_boxes (Tensor): Anchor boxes of shape (total_anchors, 4).
        image_size (tuple): Size of the input image (height, width).
        batch_size (int): Batch size.
        nms_thresh (float, optional): Non-Maximum Suppression threshold. Default is 0.7.
        pre_nms_top_n (int, optional): Number of top proposals before NMS. Default is 6000.
        post_nms_top_n (int, optional): Number of top proposals after NMS. Default is 300.
        debug (bool, optional): If True, prints debug information. Default is False.

    Returns:
        Tensor: Generated proposals of shape (num_proposals, 5), where the first column is the batch index.
    """
    device = rpn_logits.device

    if debug:
        print(f"RPN logits shape: {rpn_logits.shape}")
        print(f"RPN bbox_pred shape: {rpn_bbox_pred.shape}")
        print(f"Anchor boxes shape: {anchor_boxes.shape}")
    
    logits = rpn_logits.permute(0, 2, 3, 1).contiguous().view(-1, 1)
    bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous().view(-1, 4)
    anchors = anchor_boxes.view(-1, 4).to(device)

    if debug:
        print(f"Logits shape: {logits.shape}")
        print(f"Bbox_pred shape: {bbox_pred.shape}")
        print(f"Anchors shape: {anchors.shape}")
    
    scores = logits.squeeze()
    if debug:
        print(f"Scores shape: {scores.shape}")
    proposals = bbox_pred + anchors[:bbox_pred.shape[0], :]
    if debug:
        print(f"Proposals shape: {proposals.shape}")

    proposals[:, [0, 2]] = proposals[:, [0, 2]].clamp(0, image_size[1])
    proposals[:, [1, 3]] = proposals[:, [1, 3]].clamp(0, image_size[0])

    keep = ops.nms(proposals, scores, nms_thresh)
    if debug:
        print(f"Number of proposals after NMS: {len(keep)}")

    # Add batch index to the proposals
    keep = keep[:post_nms_top_n]
    batch_indices = torch.arange(batch_size, dtype=proposals.dtype, device=proposals.device).repeat_interleave(len(keep) // batch_size)
    rois = torch.cat([batch_indices[:, None], proposals[keep]], dim=1)

    return rois


class SwinTransformerObjectDetection(nn.Module):
    """
    Swin Transformer-based object detection model including a backbone, RPN, and heads for classification and regression.

    Args:
        backbone (nn.Module): Backbone network (Swin Transformer).
        rpn (nn.Module): Region Proposal Network.
        num_classes (int, optional): Number of object classes. Default is 2.
        debug (bool, optional): If True, prints debug information. Default is False.

    Inputs:
        x (Tensor): Input image tensor of shape (batch_size, in_chans, height, width).

    Outputs:
        Tuple[Tensor, Tensor, Tensor]: 
            cls_logits: Classification logits of shape (num_proposals, num_classes).
            bbox_regression: Bounding box regression predictions of shape (num_proposals, num_classes * 4).
            mask_pred: Mask predictions of shape (num_proposals, 1, height, width).
    """
    def __init__(self, backbone, rpn, num_classes=2, debug=False):
        super(SwinTransformerObjectDetection, self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.num_classes = num_classes
        self.debug = debug
        
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
        
        # Adjusted mask head to accept 768 channels
        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1)
        )

    def forward(self, x):
        if self.debug:
            print(f'Model input: {x.shape}')
        feature_pyramids = self.backbone(x)
        rpn_logits, rpn_bbox_pred = [], []
        feature_map = feature_pyramids[-1]
        if self.debug:
            print(f'RPN input: {feature_map.shape}')
        logits, bbox_pred = self.rpn(feature_map)
        rpn_logits.append(logits)
        rpn_bbox_pred.append(bbox_pred)
        
        feature_map_size = feature_map.size()[2:]
        anchor_boxes = generate_anchor_boxes(feature_map_size, n_anchors=logits.shape[1])

        rois = generate_proposals(rpn_logits[0], rpn_bbox_pred[0], anchor_boxes, x.size()[2:], x.size(0), debug=self.debug)
        
        pooled_features = roi_align(feature_pyramids[-1], rois, output_size=(7, 7), spatial_scale=1.0 / 16.0)
        if self.debug:
            print(f'Pooled features: {pooled_features.shape}')
        
        pooled_features_flat = pooled_features.view(pooled_features.size(0), -1)
        cls_logits = self.cls_head(pooled_features_flat)
        bbox_regression = self.bbox_head(pooled_features_flat)

        mask_pred = self.mask_head(pooled_features)

        if self.debug:
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

#summary(model, (3, 512, 512))
