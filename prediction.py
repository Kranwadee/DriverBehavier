from typing import List, Tuple
import torch
import torchvision.transforms as T
from PIL import Image

def pred_class(model: torch.nn.Module,
               image,
               class_names: List[str],
               image_size: Tuple[int, int] = (224, 224)):
    
    # Apply transformations to the image
    image_transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    
    # Ensure model is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Transform and prepare the image for the model
    transformed_image = image_transform(image).unsqueeze(dim=0).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation for inference
    with torch.inference_mode():
        # Make predictions
        target_image_pred = model(transformed_image)

        # Apply softmax to get probabilities
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

        # Get the predicted label
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

        # Extract the predicted class name
        predicted_class = class_names[target_image_pred_label]

        # Move the probabilities to CPU and convert to numpy
        prob = target_image_pred_probs.cpu().numpy()

    return predicted_class, prob
