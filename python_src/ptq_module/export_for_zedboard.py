import torch
import qtensor as ql
import numpy as np
from quant_model import QLinearLayer, QModel, emit_c_array, emit_c_float, tensor2str
from include.SmallCNN import load_data, transform, evaluate, SmallCNN
import pickle
import os
import shutil
from tqdm import tqdm

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)

IN_FEATURES = 50
CALIB_STEPS = 1000
TEST_STEPS = 1000

export_dir = "export_eval"
device = torch.device('cpu')

C = 3
H = 64
W = 64

model = SmallCNN()

# Load model checkpoint
model_path = '../ai_experiments/openset_classification_experiment_2025_06_22.pth'
model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
model.eval()

_, _, test_loader = load_data(transform=transform)

qlayer = QLinearLayer(0, "layer", model.network[-1]) # only the last layer
base_model = model.network[:-1]

for i in range(CALIB_STEPS):
    cx = torch.randn(1, C, H, W) 
    # cx = torch.randn(1, C, H, W) # TODO: calibrate on actual training data
    cx = base_model(cx)
    qlayer.calibrate(cx)

qlayer.quantize()

# print(f"len(test_loader): {len(test_loader.dataset)}")
# best_threshold = 3.5484
best_threshold = 3.005
evaluate(qlayer, base_model, device, test_loader, best_threshold)

transformed_images = []
labels_class = []
labels_osr = []

with torch.no_grad():
    for x, y, known in tqdm(test_loader):
        x = x.to(device)

        y_hat = base_model(x)
        transformed_images.append(y_hat.cpu().numpy())
        labels_class.append(y.cpu().numpy())
        labels_osr.append(known.cpu().numpy())

# stack the batches into single NumPy arrays
transformed_images = np.concatenate(transformed_images, axis=0)
labels_class = np.concatenate(labels_class, axis=0)
labels_osr = np.concatenate(labels_osr, axis=0)

if os.path.exists(export_dir):
    shutil.rmtree(export_dir)
os.mkdir(export_dir)

# after transformed_images and labels_class are ready …
np.save(os.path.join(export_dir, "transformed_images.npy"), transformed_images)
np.save(os.path.join(export_dir, "labels_class.npy"), labels_class)
np.save(os.path.join(export_dir, "labels_osr.npy"), labels_osr)
qlayer.export_npy(export_dir)

print("Transformed images exported successfully!")