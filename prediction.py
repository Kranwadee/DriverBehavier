from typing import List, Tuple
import torch
import torchvision.transforms as T
from PIL import Image

def pred_class(model: torch.nn.Module,
               image,
               class_names: List[str],
               image_size: Tuple[int, int] = (224, 224)):
    
    img = image

    image_transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    with torch.inference_mode():

        transformed_image = image_transform(img).unsqueeze(dim=0).float()

        target_image_pred = model(transformed_image.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    predicted_class = class_names[target_image_pred_label]
    prob = target_image_pred_probs.cpu().numpy()

    return predicted_class, prob
