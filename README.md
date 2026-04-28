## Pytorch_ZTM

This repository includes:

- **PyTorch practice notebooks** (`00` → `07`) covering fundamentals through experiment tracking.
- A small **modular image classification** project (**TinyVGG**) trained on a 3‑class dataset: **pizza / sushi / steak**.

If you’re mainly here for the scripted project, start with `going_modular/train.py`.

## Project structure

- **`going_modular/`**: the modular/script version of the CV project
  - **`train.py`**: end-to-end training script (creates dataloaders, trains, saves weights)
  - **`data_setup.py`**: dataset + dataloader creation (uses `torchvision.datasets.ImageFolder`)
  - **`model.py`**: `TinyVGG` model definition
  - **`engine.py`**: training loop (`train_step`, `test_step`, `train`)
  - **`utils.py`**: utilities (e.g., saving model weights)
  - **`prediction.py`**: predict + plot helper (`pred_plot_image`)
- **`data/`**: dataset folder (ImageFolder layout)
  - `data/pizza_sushi_steak/train/<class_name>/*.jpg`
  - `data/pizza_sushi_steak/test/<class_name>/*.jpg`
- **`models/`**: saved weights (`.pth`)
- **`runs/`**: TensorBoard event files (experiment tracking output)
- **`*.ipynb`**: PyTorch notebooks (learning/practice)
- **`helper_functions.py`**: shared utilities used in notebooks (plotting, downloading data, etc.)

## Requirements

### Software

- **Python**: 3.10+ recommended
- **Core libs**: `torch`, `torchvision`
- **Extras used in scripts/notebooks**: `tqdm`, `matplotlib`

### Install (minimal)

Create a virtual environment (recommended), then install:

```bash
pip install torch torchvision tqdm matplotlib
```

> For Windows / CUDA-specific installs, use the official PyTorch install selector to pick the right wheel.

## Dataset

The training script expects this folder structure:

- `data/pizza_sushi_steak/train/pizza/*.jpg`
- `data/pizza_sushi_steak/train/sushi/*.jpg`
- `data/pizza_sushi_steak/train/steak/*.jpg`
- `data/pizza_sushi_steak/test/pizza/*.jpg`
- `data/pizza_sushi_steak/test/sushi/*.jpg`
- `data/pizza_sushi_steak/test/steak/*.jpg`

Class names are inferred from the **folder names** via `ImageFolder`.

## Train

The training entrypoint is `going_modular/train.py`. It:

- uses `transforms.Resize((64, 64))` + `transforms.ToTensor()`
- trains `TinyVGG` for **5 epochs** (defaults in the script)
- prints train/test loss & accuracy each epoch (from `going_modular/engine.py`)
- saves weights to `models/05_pytorch_going_modular_script_mode.pth`

Run from the **project root**:

```bash
python going_modular/train.py
```

### Outputs

- **Saved weights**: `models/05_pytorch_going_modular_script_mode.pth`
- **Console logs**: per-epoch metrics + total training time
- **(Optional)** TensorBoard events under `runs/` if you used notebook tracking

## TensorBoard (optional)

If you have event files under `runs/`:

```bash
tensorboard --logdir runs
```

Then open `http://localhost:6006` in your browser.

## Predict on an image (optional)

`going_modular/prediction.py` contains a helper function (`pred_plot_image`) that:

- loads an image from disk
- applies a transform (or a default ImageNet-style normalization if you don't provide one)
- runs the model in inference mode
- plots the image with predicted label + probability

To use it, create `TinyVGG` and load the saved `state_dict` (same architecture and matching `class_names`).

Skeleton example (general guide):

```python
import torch
from torchvision import transforms

from going_modular import model as model_module
from going_modular.prediction import pred_plot_image

class_names = ["pizza", "steak", "sushi"]  # adjust if your dataset class order differs

net = model_module.TinyVGG(input_shape=3, hidden_units=10, output_shape=len(class_names))
net.load_state_dict(torch.load("models/05_pytorch_going_modular_script_mode.pth", map_location="cpu"))

pred_plot_image(
    model=net,
    image_path="data/pizza_sushi_steak/test/pizza/xxx.jpg",
    class_names=class_names,
    image_size=(64, 64),
    transform=transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((64, 64)), transforms.ToTensor()]
    ),
)
```

## Notebooks

These notebooks are a step-by-step progression through PyTorch concepts. Open them in Jupyter / VS Code.

- **`00_pytorch_fudamentals.ipynb`**: tensors, basic ops, device basics (PyTorch fundamentals)
- **`01_pytorch_workflow.ipynb`**: a typical training workflow (data → model → loss/optimizer → train/eval)
- **`02_pytorch_classification.ipynb`**: classification fundamentals (metrics, logits/softmax, training loops)
- **`03_pytorch_computer_vision.ipynb`**: computer vision basics with PyTorch/torchvision
- **`04_pytorch_custom_dataset.ipynb`**: custom datasets + dataloaders, transforms, and prediction utilities
- **`05_pytorch_going_modular.ipynb`**: refactoring notebook code into reusable modules (`going_modular/`)
- **`06_pytorch_transfre_learning.ipynb`**: transfer learning (fine-tuning a pretrained model)
- **`07_pytorch_experiment_tracking.ipynb`**: experiment tracking and logging (e.g., TensorBoard under `runs/`)

## Common issues

- **`ModuleNotFoundError: No module named 'data_setup'`**: run scripts from the **repo root**:
  - ✅ `python going_modular/train.py`
  - ❌ running from inside `going_modular/` may break imports (depending on your environment)
- **No GPU / CUDA not available**: the code falls back to CPU automatically (`cuda` if available, else `cpu`).

## License / notes

This is a learning/homework-style repository. If you plan to reuse parts, double-check dataset licensing and attribution.

---

If you want, I can also add a **“Report”** section (final metrics, plots, and a short discussion) tailored to your course requirements.

**Note**: The `going_modular/` folder was created by modularizing the work from `05_pytorch_going_modular.ipynb`.

