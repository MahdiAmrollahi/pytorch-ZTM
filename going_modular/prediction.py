
from torchvision.io import read_image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch

device = 'cuda' if torch.cuda.is_available() else "cpu"

def pred_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: list[str],
    image_size: tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
    device: torch.device = device,
):

    img = read_image(image_path)

    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    model.to(device)

    model.eval()
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(0)

        target_image_pred = model(transformed_image.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    target_iamge_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    plt.figure()
    plt.imshow(img.permute(1, 2, 0))
    plt.title(
        f"Pred: {class_names[target_iamge_pred_label]} | probs: {target_image_pred_probs.max():.3f}"
    )
    plt.axis("off")
