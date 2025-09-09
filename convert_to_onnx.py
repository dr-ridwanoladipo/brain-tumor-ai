"""
Clinical-Grade Brain Tumor Segmentation - ONNX Export
by Ridwan Oladipo, MD | AI Specialist

Professional nnU-Net 2025 ONNX conversion for production deployment
Targeting WT Dice ≥ 90% and BraTS Avg ≥ 80%
"""

import torch
import torch.nn as nn

# Model Architecture (inference-optimized)
def conv_block(in_f, out_f):
    return nn.Sequential(
        nn.Conv3d(in_f, out_f, 3, padding=1, bias=False),
        nn.InstanceNorm3d(out_f),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Conv3d(out_f, out_f, 3, padding=1, bias=False),
        nn.InstanceNorm3d(out_f),
        nn.LeakyReLU(0.01, inplace=True)
    )

def down_block(in_f, out_f):
    return nn.Sequential(
        nn.Conv3d(in_f, out_f, 3, stride=2, padding=1, bias=False),
        nn.InstanceNorm3d(out_f),
        nn.LeakyReLU(0.01, inplace=True)
    )

def up_block(in_f, out_f):
    return nn.Sequential(
        nn.ConvTranspose3d(in_f, out_f, 2, stride=2, bias=False),
        nn.InstanceNorm3d(out_f),
        nn.LeakyReLU(0.01, inplace=True)
    )

class nnUNet2025_ONNX(nn.Module):
    """nnU-Net 2025 for inference (main output only)"""
    def __init__(self, in_channels, out_channels, base_filters):
        super().__init__()
        # Encoder
        self.enc1 = conv_block(in_channels, base_filters)
        self.down1 = down_block(base_filters, base_filters * 2)
        self.enc2 = conv_block(base_filters * 2, base_filters * 2)
        self.down2 = down_block(base_filters * 2, base_filters * 4)
        self.enc3 = conv_block(base_filters * 4, base_filters * 4)
        self.down3 = down_block(base_filters * 4, base_filters * 8)
        self.enc4 = conv_block(base_filters * 8, base_filters * 8)
        self.down4 = down_block(base_filters * 8, base_filters * 16)

        # Bottleneck
        self.bottleneck = conv_block(base_filters * 16, base_filters * 16)

        # Decoder
        self.up4 = up_block(base_filters * 16, base_filters * 8)
        self.dec4 = conv_block(base_filters * 16, base_filters * 8)
        self.up3 = up_block(base_filters * 8, base_filters * 4)
        self.dec3 = conv_block(base_filters * 8, base_filters * 4)
        self.up2 = up_block(base_filters * 4, base_filters * 2)
        self.dec2 = conv_block(base_filters * 4, base_filters * 2)
        self.up1 = up_block(base_filters * 2, base_filters)
        self.dec1 = conv_block(base_filters * 2, base_filters)

        # Main output head only
        self.out_conv = nn.Conv3d(base_filters, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        b = self.bottleneck(self.down4(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)

# ONNX Conversion Configuration
model_path = "brain-tumor-eval/best_model.pth"
onnx_path = "brain-tumor-eval/brain-model/best_model.onnx"

in_channels = 4
out_channels = 4
base_filters = 32
patch_size = (96, 96, 96)

device = torch.device("cpu")
model = nnUNet2025_ONNX(in_channels, out_channels, base_filters)

# Load checkpoint and remove deep supervision weights
state_dict = torch.load(model_path, map_location=device)
for key in ['ds_out2.weight', 'ds_out2.bias', 'ds_out3.weight', 'ds_out3.bias']:
    state_dict.pop(key, None)

# Load model weights
model.load_state_dict(state_dict, strict=True)
model.eval()

# Create dummy input for export
dummy_input = torch.randn(1, in_channels, *patch_size, device=device)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['main_out'],
    dynamic_axes={'input': {0: 'batch'}, 'main_out': {0: 'batch'}}
)

# Validation
import onnx
import onnxruntime as ort

# Load and validate ONNX model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

# Test inference with ONNX Runtime
ort_session = ort.InferenceSession(onnx_path)
onnx_inputs = {'input': dummy_input.numpy()}
onnx_outs = ort_session.run(None, onnx_inputs)

print(f"Exported and validated ONNX model: {onnx_path}")
print(f"Output shape: {onnx_outs[0].shape}")  # Expect (1, 4, 96, 96, 96)