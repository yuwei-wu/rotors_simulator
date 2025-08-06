import torch
from torch import nn
import torch.nn.functional as F
from ultralytics import YOLO
from tqdm import tqdm

def load_yolo_model(yolo_model_name, device):
    # Load the YOLO model
    # Set the shared instance of YOLO model in each client model and global
    yolo_model = YOLO(yolo_model_name)
    yolo_model.eval()
    yolo_model = yolo_model.to(device)
    # Freeze YOLO model
    for param in yolo_model.parameters():
        param.requires_grad = False

    return yolo_model


def yolo_detect(yolo_model, x_images, n_target, device):

    # Image input shape to YOLO: (bsz * win_size, 3, 640, 640)
    img_results = yolo_model.predict(source=x_images.view(-1, 3, 640, 640), verbose=False)

    # Find the detection results
    img_xywhn, detect_masks = [], []
    for result in img_results:
        detections, detect_mask = sort_detections(result, n_target, device)  
        # Output shape for detections: [n_target, 4]
        # Output shape for detect_mask: [n_target]
        
        img_xywhn.append(detections) 
        detect_masks.append(detect_mask)
    
    return torch.stack(img_xywhn), torch.stack(detect_masks)  # Return shape: [bsz * win_size, n_target, 4]


def sort_detections(result, n_target, device):
    # Given a detection result, sort the detections based on the number of detection
    n_detections = len(result.boxes)
    detections = torch.zeros((n_target, 4), device=device)
    detect_mask = torch.zeros((n_target), dtype=torch.bool, device=device)
    # print('n_detections', n_detections)

    if n_detections == 0:
        #print('0 detections found')
        return detections, detect_mask
    
    cls_index = result.boxes.cls.int()  # Get the class indices of the detections
    if n_detections <= n_target:
        # Fill in the detecting tensor with the xywhn values
        detections[cls_index] = result.boxes.xywhn  # Fill the detections tensor with the xywhn values
        detect_mask[cls_index] = 1  # Mark the detected targets

    else: # Redundant detections exist
        # The output boxes are automatically sorted by confidence score in ultralytics YOLO
        # Fill the detections tensor with the sorted boxes
        #print(n_detections, 'detections found, but only', n_target, 'targets are needed')
        for cls_id in range(n_target):
            # Get indices where cls_id is found in sorted_cls
            indices = torch.nonzero(cls_index == cls_id, as_tuple=False)

            # Get the first index if it exists, and move it to CPU
            first_index = indices[0].item() if indices.numel() > 0 else -1
            #print('detections for cls_id', cls_id, 'first_index', first_index)

            # Fill the detections tensor with the first detection of the cls_id
            if first_index > -1:
                detections[cls_id] = result.boxes.xywhn[first_index]
                detect_mask[cls_id] = 1  # Mark the detected targets
    
    # print(detections)

    return detections, detect_mask


def load_traj_model(win_size, pred_win_size, target_num, device, ckpt=None):
    model = TransformerTrajectoryPredictor(
            win_size,
            pred_win_size,
            target_num,
            device,
        )
    model = model.to(device)
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt))
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


def traj_pred(model, odom_data, bbox_data, target_num, win_size, pred_win_size, device):
    # Shape of odom_data: (1, win_size * 12)
    # Shape of bbox_data: (1, target_num, win_size * 4)
    # Move the data to the device
    drone_odometry = odom_data.to(device)
    yolo_detect_data = bbox_data.permute(1, 0, 2) # (n_target, 1, win_size * 4)
    yolo_detect_data = yolo_detect_data.reshape(target_num, -1, win_size, 4).to(device) # (n_target, 1, win_size, 4)

    # Get the new reference point for each sample
    drone_odometry = drone_odometry.reshape(-1, win_size, 12)  # Shape: [1, win_size, 12]
    new_refer_point = drone_odometry[:, -1, :2].clone()  # Shape: [1, 2]
    #print('new reference point', new_refer_point)

    # Convert the drone odometry to the new reference point
    drone_odometry[:, :, :2] -= new_refer_point.unsqueeze(1)  # Subtract the new reference point from the first two columns
    # drone_odometry Shape: [1, win_size, 12]

    out, mask = model(drone_odometry, yolo_detect_data)
    # print('out shape:', out.shape) # (n_target, 1, pred_win_size * 2)
    out = out.reshape(target_num, -1, pred_win_size, 2)
    out += new_refer_point.unsqueeze(0).unsqueeze(2)  # Add the new reference point to the predicted trajectory
    out = out.squeeze(1)  # Remove the batch dimension, shape: (n_target, pred_win_size, 2)
    
    return out.cpu().numpy()  # Convert to numpy for further processing 


class TransformerTrajectoryPredictor(nn.Module):
    def __init__(self, win_size, pred_win_size, n_target, device):
        super(TransformerTrajectoryPredictor, self).__init__()
        
        self.win_size = win_size
        self.pred_win_size = pred_win_size
        self.n_target = n_target
        #self.yolo_model = yolo_model # Only client model will share one YOLO model
        self.device = device

        # Note: the parameters here need to be mannually adjusted 
        # according to the best performed model
        self.transformer = TrajectoryTransformer(d_model=16, 
                                                 nhead=4,
                                                 num_encoder_layers=2,
                                                 num_decoder_layers=2,
                                                 dim_feedforward=32,
                                                 dropout=0.1,
                                                 pred_window_size=pred_win_size)

    
    def forward(self, x_drone_odometry, x_xywhn):
        # x_drone_odometry: [bsz, win_size, 12]
        # x_xywhn: [n_target, bsz, win_size, 4]
        bsz = x_drone_odometry.shape[0]  # Batch size

        # Step 1: Get the mask of shape (n_target, bsz)
        img_mask = x_xywhn.view(self.n_target, bsz, -1).sum(-1) > 0 # Shape: (n_target, bsz)
        # print('image mask of all targets', img_mask)  # (n_target, bsz)
        
        # Step 2: Prepare odometry and image tensor data
        # Repeat odometry for each target → shape: (n_target, bsz, win_size, 12)
        x_drone_odometry_expanded = x_drone_odometry.unsqueeze(0).expand(self.n_target, -1, -1, -1)

        # Step 3: Concatenate along last dim → shape: (n_target, bsz, win_size * 16)
        combined = torch.cat([x_xywhn, x_drone_odometry_expanded], dim=-1) # Shape (n_target, bsz, win_size, 16)
        combined = combined.permute(1, 2, 0, 3)  # Change to (bsz, win_size, n_target, 16)

        # Step 5: Run the MLP on the combined features for predicted trajectory
        # Transformer expects input of shape (bsz, win_size, n_target, 16)
        pred_trajectory = self.transformer(combined)  # Shape: (bsz, n_target, pred_win_size, 2)
        #print('predicted trajectory shape ', pred_trajectory.shape) # [(n_target, bsz, pred_win_size * 2]

        return pred_trajectory, img_mask
    

    def train_one_epoch(self, dataloader, optimizer, client_id, epoch):
        self.transformer.train()
        
        all_train_loss_per_target = torch.zeros(self.n_target).to(self.device)
        all_mask_per_target = torch.zeros(self.n_target).to(self.device)
        for _, _, drone_odometry, yolo_detect, target_pos in tqdm(dataloader, desc=f'Client {client_id} Epoch {epoch}'):
            # drone_odometry: [bsz, win_size * 12]
            # yolo_detect: [bsz, win_size * 4, n_target]
            # target_pos: [bsz, pred_win_size * 2, n_target]
            
            # Move the data to the device
            drone_odometry = drone_odometry.to(self.device)
            yolo_detect = yolo_detect.permute(2, 0, 1) # (n_target, bsz, win_size * 4)
            yolo_detect = yolo_detect.reshape(self.n_target, -1, self.win_size, 4).to(self.device) # (n_target, bsz, win_size, 4)
            target_pos = target_pos.permute(2, 0, 1).to(self.device) # (n_target, bsz, pred_win_size * 2)

            # Get the new reference point for each sample
            drone_odometry = drone_odometry.reshape(-1, self.win_size, 12)  # Shape: [bsz, win_size, 12]
            new_refer_point = drone_odometry[:, -1, :2].clone()  # Shape: [bsz, 2]
            #print('new reference point', new_refer_point)

            # Convert the drone odometry to the new reference point
            drone_odometry[:, :, :2] -= new_refer_point.unsqueeze(1)  # Subtract the new reference point from the first two columns
            # drone_odometry Shape: [bsz, win_size, 12]

            # Convert the target position to the new reference point
            target_pos = target_pos.reshape(self.n_target, -1, self.pred_win_size, 2)  # Shape: (n_target, bsz, pred_win_size, 2)
            target_pos -= new_refer_point.unsqueeze(0).unsqueeze(2)  # Subtract the new reference point from the first two columns
            target_pos = target_pos.reshape(self.n_target, -1, self.pred_win_size*2) # Shape: (n_target, bsz, pred_win_size, 2)

            optimizer.zero_grad()
            out, mask = self(drone_odometry, yolo_detect)
            # Use the most commonly used MSE loss
            loss = F.mse_loss(out, target_pos, reduction='none')  # Shape: [target_num, bsz, pred_win_size * 2]
            masked_loss = loss.sum(-1) * mask  # Shape: [target_num, bsz]
            train_loss_per_target = masked_loss.sum(-1)  # Shape: [target_num]
            mask_per_target = mask.sum(-1)  # Shape: [target_num]

            #print('train_loss_per_target', train_loss_per_target)
            #print('mask_per_target', mask_per_target)

            # Note: different from evaluation, the mask here can have all zeros for a target
            # Therefore we use safe_div: when mask is zero, train loss is also zero; otherwise, do normal division
            # In this way, we ensure the gradient is zero when there is no detection for a target
            safe_div = torch.where(
                mask_per_target > 0,
                train_loss_per_target / (mask_per_target + 1e-6),
                torch.zeros_like(train_loss_per_target)
            )
            train_loss = safe_div.sum()
            #print('safe_div', safe_div)
            #print('train loss', train_loss)
            train_loss.backward()
            optimizer.step()
            
            # Append to all losses for logging purposes
            all_train_loss_per_target += train_loss_per_target.detach()
            all_mask_per_target += mask_per_target.detach()

        # Obtain the final training loss for logging
        safe_div = torch.where(
                all_mask_per_target > 0,
                all_train_loss_per_target / (all_mask_per_target + 1e-6),
                torch.zeros_like(all_train_loss_per_target)
            )
        train_loss_per_target = safe_div.cpu().numpy()
        # Different from training, in logging we compute mean loss over all valid targets
        # Valid targets => those with non-zero losses
        train_loss = float(train_loss_per_target[train_loss_per_target > .0].mean())
        
        return train_loss / len(dataloader.dataset), train_loss_per_target / len(dataloader.dataset)  # Return for logging
    
    
    def evaluate(self, dataloader):
        self.transformer.eval()

        test_loss_per_target = torch.zeros(self.n_target).to(self.device)
        mask_per_target = torch.zeros(self.n_target).to(self.device)
        with torch.no_grad():
            for _, _, drone_odometry, yolo_detect, target_pos in tqdm(dataloader, desc='Evaluating'):
                # drone_odometry: [bsz, win_size * 12]
                # yolo_detect: [bsz, win_size * 4, n_target]
                # target_pos: [bsz, pred_win_size * 2, n_target]

                # Move the data to the device
                drone_odometry = drone_odometry.to(self.device)
                yolo_detect = yolo_detect.permute(2, 0, 1) # (n_target, bsz, win_size * 4)
                yolo_detect = yolo_detect.reshape(self.n_target, -1, self.win_size, 4).to(self.device) # (n_target, bsz, win_size, 4)
                target_pos = target_pos.permute(2, 0, 1).to(self.device) # (n_target, bsz, pred_win_size * 2)

                # Get the new reference point for each sample
                drone_odometry = drone_odometry.reshape(-1, self.win_size, 12)  # Shape: [bsz, win_size, 12]
                new_refer_point = drone_odometry[:, -1, :2].clone()  # Shape: [bsz, 2]
                #print('new reference point', new_refer_point)

                # Convert the drone odometry to the new reference point
                drone_odometry[:, :, :2] -= new_refer_point.unsqueeze(1)  # Subtract the new reference point from the first two columns
                # drone_odometry Shape: [bsz, win_size, 12]

                # Convert the target position to the new reference point
                target_pos = target_pos.reshape(self.n_target, -1, self.pred_win_size, 2)  # Shape: (n_target, bsz, pred_win_size, 2)
                target_pos -= new_refer_point.unsqueeze(0).unsqueeze(2)  # Subtract the new reference point from the first two columns
                target_pos = target_pos.reshape(self.n_target, -1, self.pred_win_size*2) # Shape: (n_target, bsz, pred_win_size, 2)

                out, mask = self(drone_odometry, yolo_detect)
                # Use the most commonly used MSE loss
                loss = F.mse_loss(out, target_pos, reduction='none')  # Shape: [target_num, bsz, pred_win_size * 2]
                masked_loss = loss.sum(-1) * mask  # Shape: [target_num, bsz]
                test_loss_per_target += masked_loss.sum(-1)  # Shape: [target_num]
                mask_per_target += mask.sum(-1)  # Shape: [target_num]

        test_loss_per_target = test_loss_per_target / (mask_per_target + 1e-6)  # Average the loss over all detections per target
        test_loss_per_target = test_loss_per_target.cpu().numpy()
        # We compute mean loss over all valid targets
        # Valid targets => those with non-zero losses
        test_loss = float(test_loss_per_target[test_loss_per_target > .0].mean())

        return test_loss / len(dataloader.dataset), test_loss_per_target / len(dataloader.dataset)  # Return for logging


class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim=16, d_model=32, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=64, dropout=0.1, pred_window_size=20):
        super(TrajectoryTransformer, self).__init__()

        self.d_model = d_model
        self.pred_window_size = pred_window_size

        # Embed (x, y) → d_model
        self.input_embed = nn.Linear(input_dim, d_model)

        # Positional encoding for input sequence (past)
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Learnable target embeddings (optional)
        # self.target_embed = nn.Embedding(n_target, d_model)

        # Decoder: we use a learned query sequence as the future trajectory placeholder
        self.query_embed = nn.Parameter(torch.randn(pred_window_size, d_model))

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Project back to (x, y)
        self.output_proj = nn.Linear(d_model, 2)

    def forward(self, x):
        # x: (batch_size, window_size, n_target, input_dim)
        batch_size, window_size, n_target, input_dim = x.shape

        # Reshape to (batch_size * n_target, window_size, input_dim)
        x = x.permute(0, 2, 1, 3).reshape(-1, window_size, input_dim)

        # Embed: (B*n_target, window_size, d_model)
        x = self.input_embed(x)
        x = self.positional_encoding(x)

        # Transformer expects (seq_len, batch, d_model)
        x = x.permute(1, 0, 2)

        # Encode
        memory = self.encoder(x)  # (window_size, B*n_target, d_model)

        # Prepare target query for decoder
        tgt = self.query_embed.unsqueeze(1).repeat(1, memory.shape[1], 1)  # (pred_window_size, B*n_target, d_model)

        # Decode
        output = self.decoder(tgt, memory)  # (pred_window_size, B*n_target, d_model)

        # Project back to (x, y)
        output = self.output_proj(output)  # (pred_window_size, B*n_target, 2)

        # Reshape to (batch_size, n_target, pred_window_size, 2)
        output = output.permute(1, 0, 2).reshape(batch_size, n_target, self.pred_window_size, 2)
        # Reshape to (n_target, batch_size, pred_window_size * 2)
        output = output.permute(1, 0, 2, 3).reshape(n_target, batch_size, self.pred_window_size * 2)

        return output
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Even
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x
