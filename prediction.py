from typing import List, Tuple
from PIL import Image
import torch
import torchvision.transforms as transforms

def pred_class(model: torch.nn.Module,
               image: Image.Image,
               class_names: List[str],
               image_size: Tuple[int, int] = (224, 224),
               transform: transforms = None,
               device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> Tuple[str, float]:

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    ### Predict on image ###

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.no_grad():  # Use torch.no_grad() for older versions of PyTorch
        # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(image).unsqueeze(dim=0).to(device)

        # 7. Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image)

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1).item()

    classname = class_names[target_image_pred_label]
    prob = target_image_pred_probs.max().cpu().numpy()

    return classname, prob
