from typing import List, Tuple
from PIL import Image
import torch
from torchvision import transforms

def pred_class(model: torch.nn.Module,
               image: Image.Image,
               class_names: List[str],
               image_size: Tuple[int, int] = (224, 224),
               transform: transforms.Compose = None,
               device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> Tuple[str, float]:

    # Create transformation for image (if one doesn't exist)
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # Transform the image and add an extra dimension
    transformed_image = transform(image).unsqueeze(dim=0).to(device)

    # Convert image to float16 if model expects half precision
    if next(model.parameters()).dtype == torch.float16:
        transformed_image = transformed_image.half()

    # Ensure the model is in evaluation mode and on the correct device
    model.eval()
    model.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(transformed_image)
    
    # Apply softmax to get prediction probabilities
    probabilities = torch.softmax(output, dim=1)

    # Get the predicted class index and probability
    predicted_index = torch.argmax(probabilities, dim=1).item()
    predicted_class = class_names[predicted_index]
    predicted_prob = probabilities[0, predicted_index].item()

    return predicted_class, predicted_prob
