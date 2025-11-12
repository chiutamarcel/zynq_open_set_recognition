#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import torchvision
from torchvision import transforms
import torchvision.models as models
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from quant_model import emit_c_array, emit_c_float
import os
from include.eurosat import *
import numpy as np
import time

BATCH_SIZE = 32
IMG_DIM_X = 64
IMG_DIM_Y = 64

BEST_THRESHOLD = 3.5834

label_map = {
    0: "AnnualCrop",
    1: "Forest",
    2: "HerbaceousVegetation",
    3: "Highway",
    4: "Industrial",
    5: "Pasture",
    6: "PermanentCrop",
    7: "Residential",
    8: "River",
    9: "SeaLake"
}

known_labels = {
    0, 1, 3, 4, 5, 7, 9
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop((IMG_DIM_X,IMG_DIM_Y)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_data(transform, dir="dataset/eurosat"):
    train_csv = os.path.join(dir, "train_filtered.csv")
    val_csv = os.path.join(dir, "validation.csv")
    test_csv = os.path.join(dir, "test.csv")

    train_set = EuroSAT(csv_file=train_csv, root_dir=dir, transform=transform)
    val_set = EuroSAT(csv_file=val_csv, root_dir=dir, transform=transform)
    test_set = EuroSAT(csv_file=test_csv, root_dir=dir, transform=transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)  # No need to shuffle test set

    return train_loader, val_loader, test_loader

def export_test_vectors(export_dir, x, y):
    filepath = os.path.join(export_dir, 'test_vectors.h')
    with open(filepath, 'w') as f:
        emit_c_array(f, "test_x", x, "float")
        emit_c_array(f, 'test_y', y, "float")


def compute_per_class_f1_precision_recall_support(y_pred, y_true, the_class):
    tp = np.sum((y_true == the_class) & (y_pred == the_class))
    fp = np.sum((y_true != the_class) & (y_pred == the_class))
    fn = np.sum((y_true == the_class) & (y_pred != the_class))
    support = np.sum((y_true == the_class))

    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    
    return precision, recall, f1_score, support

def compute_weighted_f1(y_pred, y_true):
  classes = np.unique(y_true)
  classes2 = np.unique(y_pred)

  if ((classes != classes2).any()):
    raise RuntimeError("Number of classes doesn't match in compute_weighted_f1 between labels and predicted output")

  weighted_f1 = 0
  total_sup = 0

  for c in classes:
    _, _, f1, sup = compute_per_class_f1_precision_recall_support(y_pred, y_true, c)
    weighted_f1 += sup * f1
    total_sup += sup

  return weighted_f1 / float(total_sup)
    

def compute_metrics(labels_class, labels_osr, logits, mls_thresh):
    logits = np.array(logits)
    
    mls = np.max(logits, axis=1)
    y_pred_osr = (mls > mls_thresh).astype(int)

    y_pred_class = np.argmax(logits, axis=1)
    mask = (labels_osr == 1) # compare for known classes only
    labels_class_filtered = labels_class[mask]
    y_pred_filtered = y_pred_class[mask]

    f1_score_osr = compute_weighted_f1(y_pred_osr, labels_osr)
    print(f"f1_score (osr): {f1_score_osr:.4f}")

    f1_score_class = compute_weighted_f1(y_pred_filtered, labels_class_filtered)
    print(f"f1_score (classification) = {f1_score_class:.4f}")

def evaluate(qmodel, base_model, device, loader, best_threshold):
  base_model.eval()

  known_true = []
  known_pred = []
  y_true = []
  y_pred = []

  start_time = time.time()

  # Evaluate test dataset
  with torch.no_grad():
    for x, y, known in tqdm(loader):
      x = x.to(device)

      y_hat = base_model(x)
      y_hat = qmodel.forward(y_hat)

      mls = np.max(y_hat.cpu().numpy(), axis=1)

      known_true.extend(known.numpy())
      known_pred.extend((mls > best_threshold).astype(int))
      y_true.extend(y.cpu())
      y_pred.extend(np.argmax(y_hat.cpu(), axis=1))

  end_time = time.time()
  elapsed_time = end_time - start_time
  print(f"Evaluation elapsed time: {elapsed_time:.4f}")

  known_true = np.array(known_true)
  known_pred = np.array(known_pred)
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)

  # filter class prediction where class is unknown
  mask = (known_true != 0)
  y_true_filtered = y_true[mask]
  y_pred_filtered = y_pred[mask]

  # compute_metrics(known_pred, known_true, 1)
  f1_score_osr = compute_weighted_f1(known_pred, known_true)
  print(f"f1_score (osr) = {f1_score_osr:.4f}")

  f1_score_class = compute_weighted_f1(y_pred_filtered, y_true_filtered)
  print(f"f1_score (classification) = {f1_score_class:.4f}")

  # TODO: uncomment this
#   report = classification_report(known_true, known_pred, target_names=['unkown', 'known'])
#   print("Closed-set report")
#   print(report)

#   target_names = [label_map[i] for i in known_labels]

#   report = classification_report(y_true_filtered, y_pred_filtered, target_names=target_names)
#   print("Open-set report")
#   print(report)

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SmallCNN, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(4,4),
            nn.Flatten(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        return self.network(x)

