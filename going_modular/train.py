
import os 
from timeit import default_timer as timer

import torch
from torchvision import transforms
import data_setup, engine, model, utils

# Setup hyperprameters
EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup directories
train_dir = 'data/pizza_sushi_steak/train/'
test_dir = 'data/pizza_sushi_steak/test/'

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

# Create DataLoader and get class_names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloader(
                                train_dir=train_dir,
                                test_dir=test_dir,
                                transform=data_transform,
                                batch_size=BATCH_SIZE)

# Create model
model = model.TinyVGG(input_shape=3,
                        hidden_units=HIDDEN_UNITS,
                        output_shape=len(class_names)).to(device)

# Setup loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),
                                lr=LEARNING_RATE)

# Start the timer
start_time = timer()

# Start training
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=EPOCHS,
             device=device)

# End the timer 
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds.")

# Save model
utils.save_model(model=model,
                 target_dir='models',
                 model_name='05_pytorch_going_modular_script_mode.pth')
