
from pathlib import Path
import torch

def save_model(model: torch.nn.Module,
                target_dir: str,
                model_name: str):

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                            exist_ok=True)

    assert model_name.endswith('.pth') or model_name.endswith('.pt')
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.parameters(), f=model_save_path)

