import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F



def load_traj_model(win_size, pred_win_size, target_num, device, ckpt=None):
    model = MultiRobotTrajPredictor(
        win_size,
        pred_win_size,
        target_num,
        device
    )
    model = model.to(device)
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt))
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


def traj_pred(model, odom_data, image_data, target_num, win_size, pred_win_size, device):
    # Shape of odom data: (1, target_num, win_size * 12)
    # Shape of image data: (1, robot_num, win_size, 3, 640, 640)
    # Move the data to the device
    bsz = odom_data.shape[0]
    n_robot = odom_data.shape[1]

    # Move the data to the device
    drone_odometry = odom_data.to(device)
    images = image_data.to(device)

    # Get the new reference point for each sample
    # The reference point is x,y position of the latest sample on robot 0
    drone_odometry = drone_odometry.reshape(-1, n_robot, win_size, 12)  # Shape: [bsz, n_client, win_size, 12]
    new_refer_point = drone_odometry[:, 0, -1, :2].clone()  # Shape: [bsz, 2]
    # print('new reference point', new_refer_point)

    # Convert the drone odometry to the new reference point
    # Subtract the new reference point from the first two columns
    drone_odometry[:, :, :, :2] -= new_refer_point.unsqueeze(1).unsqueeze(1)
    # drone_odometry Shape: [bsz, n_client, win_size, 12]

    out = model.model(images, drone_odometry) # Shape: [bsz, pred_win_size, 2, n_target]
    out = out.permute(3, 0, 1, 2).contiguous() # Shape: [n_target, bsz, pred_win_size, 2]
    out = out.reshape(target_num, bsz, pred_win_size*2) # Shape: [n_target, bsz, pred_win_size*2]


    # Add the predicted trajectory with the reference point
    # print('out shape:', out.shape) # (n_target, bsz, pred_win_size * 2)
    out = out.reshape(target_num, -1, pred_win_size, 2)
    # print('out before', out.shape, out)
    # out[0, 0, 0, 0] -= 5.0  # TEMP FIX to let the centralized baseline work on 2-target, 2-drone case
    out += new_refer_point.unsqueeze(0).unsqueeze(2)  # Add the new reference point to the predicted trajectory
    # print('out after', out)
    out = out.reshape(target_num, -1, pred_win_size * 2)  # Shape: (n_target, bsz, pred_win_size * 2)

    return out.cpu().numpy()


# -------------------------
# Positional embeddings
# -------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # not a parameter

    def forward(self, x: torch.Tensor, positions: torch.Tensor):
        """
        x: [B, N, d_model]
        positions: [B, N] integer positions (e.g., time indices)
        """
        # gather sinusoidal rows
        pe = self.pe[positions]  # [B, N, d_model]
        return x + pe


# -------------------------
# Encoders
# -------------------------
class SimpleCNNEncoder(nn.Module):
    """Lightweight CNN that turns a single image into one feature vector."""
    def __init__(self, in_ch: int, d_model: int):
        super().__init__()
        width = 64
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, kernel_size=7, stride=2, padding=3),  # 320x320
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),  # 160x160

            nn.Conv2d(width, width*2, 3, stride=2, padding=1),  # 80x80
            nn.BatchNorm2d(width*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(width*2, width*4, 3, stride=2, padding=1),  # 40x40
            nn.BatchNorm2d(width*4),
            nn.ReLU(inplace=True),

            nn.Conv2d(width*4, width*4, 3, stride=2, padding=1),  # 20x20
            nn.BatchNorm2d(width*4),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)  # [B, width*4, 1, 1]
        )
        self.proj = nn.Linear(width*4, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        returns: [B, d_model]
        """
        feat = self.net(x).flatten(1)  # [B, width*4]
        return self.proj(feat)         # [B, d_model]


class OdomEncoder(nn.Module):
    def __init__(self, odom_dim: int, d_model: int):
        super().__init__()
        hidden = max(128, d_model // 2)
        self.mlp = nn.Sequential(
            nn.Linear(odom_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, odom_dim]
        returns: [B, d_model]
        """
        return self.mlp(x)


class Fusion(nn.Module):
    """Fuse image and odom embeddings (concat + linear)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.lin = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, img_emb: torch.Tensor, odom_emb: torch.Tensor) -> torch.Tensor:
        z = torch.cat([img_emb, odom_emb], dim=-1)
        z = self.lin(z)
        return self.norm(F.relu(z))


# -------------------------
# Trajectory Transformer
# -------------------------
class MultiRobotTrajTransformer(nn.Module):
    def __init__(
        self,
        image_channels: int = 4,
        odom_dim: int = 12,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        n_robot_max: int = 32,       # for robot embedding table
        t_obs_max: int = 256,        # for sinusoidal PE range checks
        n_target: int = 10,
        t_pred: int = 20
    ):
        super().__init__()

        self.d_model = d_model
        self.n_target = n_target
        self.t_pred = t_pred

        # Encoders
        self.img_enc = SimpleCNNEncoder(image_channels, d_model)
        self.odom_enc = OdomEncoder(odom_dim, d_model)
        self.fuse = Fusion(d_model)

        # Embeddings for robot ids and sinusoidal for time
        self.robot_embed = nn.Embedding(n_robot_max, d_model)
        self.time_pe = SinusoidalPositionalEncoding(d_model, max_len=t_obs_max + 1024)

        # Encoder/Decoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Query embeddings for (future step, target id)
        self.future_embed = nn.Embedding(t_pred, d_model)
        self.target_embed = nn.Embedding(n_target, d_model)

        # Regress head -> (x, y)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 2)
        )

    def forward(
        self,
        images: torch.Tensor,          # [B, R, T_obs, C, H, W]
        odom: torch.Tensor,            # [B, R, T_obs, odom_dim]
        robot_ids: Optional[torch.Tensor] = None,  # [B, R] int ids (0..n_robot_max-1)
        time_idx: Optional[torch.Tensor] = None    # [B, T_obs] int time positions (0..)
    ) -> torch.Tensor:
        B, R, T_obs, C, H, W = images.shape
        _, _, _, od = odom.shape

        # Flatten batch of tokens: (B*R*T_obs)
        imgs_flat = images.reshape(B*R*T_obs, C, H, W)
        odom_flat = odom.reshape(B*R*T_obs, od)

        img_tok = self.img_enc(imgs_flat)     # [B*R*T_obs, d]
        odom_tok = self.odom_enc(odom_flat)   # [B*R*T_obs, d]
        z = self.fuse(img_tok, odom_tok)      # [B*R*T_obs, d]

        # Unflatten into [B, R, T_obs, d]
        z = z.view(B, R, T_obs, self.d_model)

        # Add robot embeddings
        if robot_ids is None:
            # default robot ids 0..R-1 (broadcast to batch)
            rid = torch.arange(R, device=images.device).unsqueeze(0).expand(B, R)  # [B, R]
        else:
            rid = robot_ids.to(images.device)  # [B, R]
        robot_tok = self.robot_embed(rid).unsqueeze(2)  # [B, R, 1, d]
        z = z + robot_tok

        # Add temporal sinusoidal PE (same for all robots at each time)
        if time_idx is None:
            tidx = torch.arange(T_obs, device=images.device).unsqueeze(0).expand(B, T_obs)  # [B, T_obs]
        else:
            tidx = time_idx.to(images.device)  # [B, T_obs]

        # tile time positions per robot, then flatten R dimension into sequence
        z = z.view(B, R*T_obs, self.d_model)                  # [B, R*T_obs, d]
        tpos = tidx.unsqueeze(1).repeat(1, R, 1).view(B, R*T_obs)  # [B, R*T_obs]
        z = self.time_pe(z, tpos)                             # add PE

        # Encoder over all (robot,time) tokens
        mem = self.encoder(z)  # [B, R*T_obs, d]

        # Build decoder queries for all (future step, target)
        future_ids = torch.arange(self.t_pred, device=images.device)  # [T_pred]
        target_ids = torch.arange(self.n_target, device=images.device)  # [nTarget]
        F_emb = self.future_embed(future_ids)      # [T_pred, d]
        T_emb = self.target_embed(target_ids)      # [nTarget, d]

        # Combine: q[f, t] = F_emb[f] + T_emb[t]
        q = F_emb.unsqueeze(1) + T_emb.unsqueeze(0)  # [T_pred, nTarget, d]
        q = q.view(self.t_pred * self.n_target, self.d_model)  # [T_pred*nTarget, d]
        q = q.unsqueeze(0).expand(B, -1, -1).contiguous()      # [B, T_pred*nTarget, d]

        # Decoder (cross-attend to encoder memory)
        dec = self.decoder(tgt=q, memory=mem)  # [B, T_pred*nTarget, d]

        # Regress (x, y) then reshape to desired output
        xy = self.head(dec)  # [B, T_pred*nTarget, 2]
        xy = xy.view(B, self.t_pred, self.n_target, 2)  # [B, T_pred, nTarget, 2]
        xy = xy.permute(0, 1, 3, 2).contiguous()        # [B, T_pred, 2, nTarget]

        return xy


class MultiRobotTrajPredictor(nn.Module):
    def __init__(self, win_size, pred_win_size, n_target, device):
        super(MultiRobotTrajPredictor, self).__init__()
        self.win_size = win_size
        self.pred_win_size = pred_win_size
        self.n_target = n_target
        self.device = device

        self.model = MultiRobotTrajTransformer(
            image_channels=3,
            odom_dim=12,
            d_model=16,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=32,
            dropout=0.1,
            n_robot_max=32,
            t_obs_max=win_size,
            n_target=n_target,
            t_pred=pred_win_size
        ).to(device)


    