import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """
    Generator architecture based on Table 2 of the MFCL paper for CIFAR-100.
    Input: Noise vector z
    Output: Synthetic Image (3, 32, 32)
    """
    def __init__(self, z_dim=100, ngf=64, img_channels=3):
        super(Generator, self).__init__()
        self.fc = nn.Linear(z_dim, 128 * 8 * 8)
        self.bn1 = nn.BatchNorm2d(128)
        
        self.conv1 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, img_channels, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(img_channels) # Note: Table 2 lists BN at end before Tanh?
        # Usually BN is not applied on output RGB, but Table 2 says "BatchNorm(3)". 
        # We will follow standard GAN practice for stability if Table 2 is ambiguous, 
        # but the paper explicitly lists it.
        
    def forward(self, z):
        # FC -> Reshape
        x = self.fc(z)
        x = x.view(-1, 128, 8, 8)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        
        # Interpolate (Up 2x) -> Conv -> BN -> LeakyReLU
        x = F.interpolate(x, scale_factor=2)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        
        # Interpolate (Up 2x) -> Conv -> BN -> LeakyReLU
        x = F.interpolate(x, scale_factor=2)
        x = self.conv2(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)
        
        # Conv -> Tanh -> BN (Table 2 order seems: Conv, Tanh, BN)
        x = self.conv3(x)
        x = torch.tanh(x)
        x = self.bn4(x) 
        return x

def update_bn_statistics(model, data_loader, device):
    """Updates BN running mean/var of the model using provided data."""
    model.train()
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            model(images)

def get_bn_statistics(model):
    """Extracts mean and variance from all BN layers."""
    means = []
    vars = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            means.append(m.running_mean)
            vars.append(m.running_var)
    return means, vars

class BNLoss(nn.Module):
    """
    Equation (3): KL divergence between real model BN stats and synthetic BN stats.
    """
    def __init__(self):
        super(BNLoss, self).__init__()

    def forward(self, model_real, model_fake_out_stats):
        # Note: This requires hooking the model to get stats during forward pass of synthetic data
        # Or, simpler: we assume we can extract current batch stats from the model 
        # if we put it in train mode, but we need the specific batch stats, not running stats.
        
        # Implementation strategy: The paper minimizes layer-wise distance.
        # We need to hook the global model to capture input mean/var at BN layers during Generator training.
        loss = 0
        return loss # Placeholder: Actual implementation requires forward hooks (see Server implementation)