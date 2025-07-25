import torch
from ultralytics import YOLO

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
