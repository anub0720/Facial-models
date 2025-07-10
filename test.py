# test.py (FINAL VERSION)
import os
import sys
import subprocess

# ---------------------------
# Auto-install all required packages if missing
# ---------------------------
print("Enter the test folder path for Task A:")
TEST_PATH_TASK_A = input().strip()
print("Enter the test folder path for Task B:")
TEST_PATH_TASK_B = input().strip()
REQUIRED_PACKAGES = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("Pillow", "PIL"),
    ("opencv-python", "cv2"),
    ("timm", "timm"),
    ("pytorch-lightning", "pytorch_lightning"),
    ("scikit-learn", "sklearn"),
    ("tqdm", "tqdm"),
    ("numpy", "numpy")
]

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])


for pip_name, import_name in REQUIRED_PACKAGES:
    try:
        __import__(import_name)
    except ImportError:
        install(pip_name)
# Now safe to import everything else
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, accuracy_score
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, accuracy_score
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import warnings
import cv2
import pytorch_lightning as pl
from collections import Counter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# ---------------------------
# Replace these with your actual checkpoint paths
# ---------------------------
import urllib.request

MODEL_DIR = "face_detection_models"

def setup_face_detection_models():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Save targets (file paths)
    prototxt_path = os.path.join(MODEL_DIR, "deploy.prototxt")
    caffemodel_path = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

    # URLs to download from
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    caffemodel_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

    if not os.path.exists(prototxt_path):
        print("Downloading deploy.prototxt...")
        urllib.request.urlretrieve(prototxt_url, prototxt_path)

    if not os.path.exists(caffemodel_path):
        print("Downloading res10_300x300_ssd_iter_140000.caffemodel...")
        urllib.request.urlretrieve(caffemodel_url, caffemodel_path)



CHECKPOINT_PATH_TASK_A = "./Task-A/best_model_taskA.ckpt"
CHECKPOINT_PATH_TASK_B = "./Task-B/best_embedding_model_TaskB.pth"
TEST_PREPROCESSED_OUTPUT_DIR = 'preprocessed_test_faces_output'
# ---------------------------
# Ask user to enter paths at runtime
# ---------------------------

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
import timm
# --- Preprocessing Function ---
def preprocess_and_detect_face(input_folder, output_folder):
    """
    Preprocesses images, detects a single face, and saves the cropped face
    to a new folder. If no high-confidence face is found, it selects the
    detection with the highest overall confidence.
    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where preprocessed and
                              cropped face images will be saved.
    """
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")
        else:
            print(f"Output folder already exists: {output_folder}")
    except OSError as e:
        print(f"Error creating output folder '{output_folder}': {e}")
        print("Please check your permissions or the specified path.")
        return

    prototxt_path = os.path.join(MODEL_DIR, "deploy.prototxt")
    caffemodel_path = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

    if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
        print("Error: Pre-trained Caffe model files not found. Please run setup_face_detection_models().")
        return

    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    print("Pre-trained face detection model loaded successfully.")

    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist. Please check the path.")
        return

    # Iterate through gender subfolders (assuming 'male' and 'female')
    for gender in ['female', 'male']:
        current_gender_input_folder = os.path.join(input_folder, gender)
        current_gender_output_folder = os.path.join(output_folder, gender)

        if not os.path.exists(current_gender_input_folder):
            print(f"Warning: Gender input folder '{current_gender_input_folder}' not found. Skipping.")
            continue
        
        # Create output gender folder
        if not os.path.exists(current_gender_output_folder):
            os.makedirs(current_gender_output_folder)
            print(f"Created output gender folder: {current_gender_output_folder}")

        print(f"\nProcessing images in: {current_gender_input_folder}")
        for filename in tqdm(os.listdir(current_gender_input_folder), desc=f"Processing {gender} images"):
            if os.path.isdir(os.path.join(current_gender_input_folder, filename)):
                continue

            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(current_gender_input_folder, filename)

                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not read image {filename}. Skipping.")
                    continue

                original_image = image.copy()

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                normalized_gray = gray.astype(np.float32) / 255.0
                blurred_image = cv2.GaussianBlur(normalized_gray, (5, 5), 0)
                laplacian = cv2.Laplacian(blurred_image, cv2.CV_32F)
                sharpened_image = cv2.normalize(laplacian, None, 0, 1, cv2.NORM_MINMAX)
                preprocessed_image = np.uint8(sharpened_image * 255)
                preprocessed_image_bgr = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)

                h, w = preprocessed_image_bgr.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(preprocessed_image_bgr, (300, 300)), 1.0,
                                             (300, 300), (104.0, 177.0, 123.0), False, False)
                net.setInput(blob)
                detections = net.forward()

                max_confidence_above_threshold = -1
                best_bbox_above_threshold = None
                max_overall_confidence = -1
                best_overall_bbox = None

                for i in range(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    current_bbox = tuple(box.astype("int"))

                    if confidence > 0.5:
                        if confidence > max_confidence_above_threshold:
                            max_confidence_above_threshold = confidence
                            best_bbox_above_threshold = current_bbox

                    if confidence > max_overall_confidence:
                        max_overall_confidence = confidence
                        best_overall_bbox = current_bbox

                final_bbox = None
                if best_bbox_above_threshold is not None:
                    final_bbox = best_bbox_above_threshold
                elif best_overall_bbox is not None:
                    final_bbox = best_overall_bbox

                output_filename = f"preprocessed_face_{filename}"
                output_path = os.path.join(current_gender_output_folder, output_filename)

                if final_bbox:
                    (startX, startY, endX, endY) = final_bbox
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)

                    cropped_face = original_image[startY:endY, startX:endX]

                    if cropped_face.size == 0:
                        cropped_face = np.zeros((224, 224, 3), dtype=np.uint8)
                    cv2.imwrite(output_path, cropped_face)
                else:
                    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
                    cv2.imwrite(output_path, dummy_image)

# --- 0. Squeeze-and-Excitation (SE) Block Implementation ---
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        y = self.avg_pool(x.unsqueeze(-1)).squeeze(-1)
        y = self.fc(y).view(b, c)
        return x * y.expand_as(x)

# --- 0. Focal Loss Implementation ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', epsilon=1e-12, label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.epsilon = epsilon
        self.label_smoothing = label_smoothing

        if self.alpha is not None:
            if not isinstance(self.alpha, torch.Tensor):
                self.alpha = torch.tensor(self.alpha, dtype=torch.float32)

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        
        if self.label_smoothing > 0:
            smoothed_targets = torch.full_like(inputs, self.label_smoothing / (num_classes - 1))
            smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smoothed_targets = F.one_hot(targets, num_classes=num_classes).float()

        log_pt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_pt)

        pt_true_class = pt.gather(1, targets.long().unsqueeze(1)).squeeze()

        base_loss = -(smoothed_targets * log_pt).sum(dim=1)
        
        focal_term = (1 - pt_true_class).pow(self.gamma)
        loss = focal_term * base_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets.long())
            loss = alpha_t * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# --- Enhanced Supervised Contrastive Loss with Class Reweighting and Hard Negative Mining ---
class EnhancedSupConLoss(nn.Module):
    def __init__(self, temperature=0.05, base_temperature=0.07, contrast_mode='all', 
                 hard_mining_ratio=0.35, margin=0.2):
        super(EnhancedSupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode
        self.hard_mining_ratio = hard_mining_ratio
        self.margin = margin
        
    def forward(self, features, labels=None, mask=None, class_weights=None):                
        device = features.device

        if len(features.shape) < 3:
            features = features.unsqueeze(1)
        
        batch_size = features.shape[0]
        original_n_views = features.shape[1]

        if self.contrast_mode == 'one':
            if original_n_views < 2:
                raise ValueError("`contrast_mode='one'` requires at least 2 views (e.g., [bsz, 2, feature_dim])")
            anchor_feature = features[:, 0]
            contrast_feature = features[:, 1]
        elif self.contrast_mode == 'all':
            anchor_feature = features.view(-1, features.shape[-1])
            contrast_feature = anchor_feature
            
            if labels is not None:
                labels = labels.repeat_interleave(original_n_views)
                
            batch_size = anchor_feature.shape[0]
        else:
            raise ValueError('Unknown contrast mode: {}'.format(self.contrast_mode))
            
        anchor_feature = F.normalize(anchor_feature, dim=1)
        contrast_feature = F.normalize(contrast_feature, dim=1)

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        anchor_dot_contrast = anchor_dot_contrast - logits_max.detach()
        
        logits_mask = 1 - torch.eye(batch_size, device=device) if self.contrast_mode == 'all' else torch.ones(batch_size, batch_size, device=device)
        
        if mask is None: 
            mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)
        
        neg_mask = (1 - mask) * logits_mask
        
        if self.margin > 0:
            margined_neg_logits = (anchor_dot_contrast * neg_mask) - (self.margin * neg_mask)
            anchor_dot_contrast = (anchor_dot_contrast * mask) + margined_neg_logits
        
        if self.hard_mining_ratio < 1.0 and self.hard_mining_ratio > 0:
            k = int(batch_size * self.hard_mining_ratio)
            k = max(k, 1)

            current_logits_mask = mask.clone()

            for i in range(batch_size):
                valid_neg_indices = torch.where(neg_mask[i] > 0)[0]

                if len(valid_neg_indices) > 0:
                    neg_sims = anchor_dot_contrast[i, valid_neg_indices]
                    k_actual = min(k, len(valid_neg_indices))
                    if k_actual > 0:
                        _, hard_neg_local_indices = torch.topk(neg_sims, k_actual)
                        hard_neg_global_indices = valid_neg_indices[hard_neg_local_indices]
                        current_logits_mask[i, hard_neg_global_indices] = 1.0
                
            logits_mask = current_logits_mask
        
        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        positive_log_probs = log_prob * mask

        sum_positive_log_probs = positive_log_probs.sum(1)
        count_positive_pairs = mask.sum(1) + 1e-12

        if class_weights is not None and labels is not None:
            if self.contrast_mode == 'all':
                original_labels_for_weights = labels[::original_n_views]
            else:
                original_labels_for_weights = labels

            weight_values = torch.tensor([class_weights.get(label.item(), 1.0)
                                          for label in original_labels_for_weights], device=device)
            
            if self.contrast_mode == 'all':
                weight_values = weight_values.repeat_interleave(original_n_views)

            weighted_mean_log_prob_pos = (sum_positive_log_probs * weight_values) / count_positive_pairs
        else:
            weighted_mean_log_prob_pos = sum_positive_log_probs / count_positive_pairs
        
        loss = -(self.temperature / self.base_temperature) * weighted_mean_log_prob_pos
        loss = loss.mean()
        
        return loss

# --- 1. Custom Dataset for Multi-View Augmentation and Combined Images ---
class GenderDataset(Dataset):
    def __init__(self, data_dir, preprocessed_data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.preprocessed_data_dir = preprocessed_data_dir
        self.transform = transform
        self.is_train = is_train
        self.image_paths = []
        self.preprocessed_image_paths = []
        self.labels = [] # 0 for female, 1 for male (consistent mapping)
        self.class_to_idx = {'female': 0, 'male': 1}
        self.idx_to_class = {0: 'female', 1: 'male'}
        
        print(f"Loading dataset from: {data_dir} (is_train={is_train})")

        temp_image_paths = []
        temp_preprocessed_image_paths = []
        temp_labels = []

        for gender in ['female', 'male']:
            gender_path = os.path.join(data_dir, gender)
            preprocessed_gender_path = os.path.join(preprocessed_data_dir, gender)
            class_idx = self.class_to_idx[gender]
            
            if not os.path.exists(gender_path):
                print(f"Warning: Directory not found: {gender_path}. Skipping.")
                continue
            if not os.path.exists(preprocessed_gender_path):
                print(f"Warning: Directory not found: {preprocessed_gender_path}. Skipping.")
                continue

            for img_name in os.listdir(gender_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    original_img_path = os.path.join(gender_path, img_name)
                    preprocessed_img_name = f"preprocessed_face_{img_name}"
                    preprocessed_img_path = os.path.join(preprocessed_gender_path, preprocessed_img_name)

                    if os.path.exists(preprocessed_img_path):
                        temp_image_paths.append(original_img_path)
                        temp_preprocessed_image_paths.append(preprocessed_img_path)
                        temp_labels.append(class_idx)
                    else:
                        print(f"Warning: Corresponding preprocessed image not found for {original_img_path}. Skipping.")
        
        self.image_paths = temp_image_paths
        self.preprocessed_image_paths = temp_preprocessed_image_paths
        self.labels = temp_labels

        self.class_counts = Counter(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        preprocessed_img_path = self.preprocessed_image_paths[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading original image {img_path}: {e}")
            img = Image.new('RGB', (224, 224), color='black')

        try:
            preprocessed_img = Image.open(preprocessed_img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading preprocessed image {preprocessed_img_path}: {e}")
            preprocessed_img = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            if self.is_train:
                img1_original = self.transform(img)
                img2_original = self.transform(img)

                img1_processed = self.transform(preprocessed_img)
                img2_processed = self.transform(preprocessed_img)
                
                return (img1_original, img2_original, img1_processed, img2_processed), label
            else:
                img_transformed = self.transform(img)
                preprocessed_img_transformed = self.transform(preprocessed_img)
                return (img_transformed, preprocessed_img_transformed), label
        
        return (img, preprocessed_img), label

# --- 2. PyTorch Lightning DataModule ---
class GenderDataModule(pl.LightningDataModule):
    def __init__(self, train_data_dir, val_data_dir, train_preprocessed_data_dir, val_preprocessed_data_dir, 
                 batch_size=64, num_workers=4, image_size=(224, 224), test_data_dir=None, test_preprocessed_data_dir=None):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.train_preprocessed_data_dir = train_preprocessed_data_dir
        self.val_preprocessed_data_dir = val_preprocessed_data_dir
        self.test_data_dir = test_data_dir
        self.test_preprocessed_data_dir = test_preprocessed_data_dir

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomRotation(degrees=20),
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            self.normalize,
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3))
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(int(image_size[0] / 0.875)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            self.normalize
        ])

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_weights_for_loss = None
        self.class_weights_tensor = None

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = GenderDataset(self.train_data_dir, self.train_preprocessed_data_dir, transform=self.train_transform, is_train=True)
            self.val_dataset = GenderDataset(self.val_data_dir, self.val_preprocessed_data_dir, 
                                             transform=self.val_transform, is_train=False)

            total_samples = len(self.train_dataset)
            num_classes = len(self.train_dataset.class_to_idx)
            
            female_count = self.train_dataset.class_counts.get(0, 0)
            male_count = self.train_dataset.class_counts.get(1, 0)

            weight_female = total_samples / (num_classes * female_count) if female_count > 0 else 1.0
            weight_male = total_samples / (num_classes * male_count) if male_count > 0 else 1.0
            
            self.class_weights_for_loss = {0: weight_female, 1: weight_male}
            self.class_weights_tensor = torch.tensor([weight_female, weight_male], dtype=torch.float32)

            print(f"Calculated class weights for loss (Female: {self.class_weights_for_loss[0]:.2f}, Male: {self.class_weights_for_loss[1]:.2f})")
            print(f"Train dataset class counts: {self.train_dataset.class_counts}")

        if stage == 'test' or stage is None:
            if self.test_data_dir and self.test_preprocessed_data_dir:
                self.test_dataset = GenderDataset(self.test_data_dir, self.test_preprocessed_data_dir,
                                                  transform=self.val_transform, is_train=False)

    def train_dataloader(self):
        labels = self.train_dataset.labels
        sample_weights = [self.class_weights_for_loss[label] for label in labels]
        
        sample_weights = np.array(sample_weights)
        sample_weights[~np.isfinite(sample_weights)] = 1.0

        sampler = WeightedRandomSampler(
            weights=list(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
        else:
            raise RuntimeError("Test dataset not set up. Call setup('test') first.")


# --- 3. PyTorch Lightning Model with EfficientNetB3 and EnhancedSupConLoss ---
class GenderClassificationModel(pl.LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-4, weight_decay=1e-5, 
                 supcon_temp=0.07, supcon_base_temp=0.07, supcon_hard_mining_ratio=0.35, 
                 supcon_margin=0.2, class_weights_for_loss=None, class_weights_tensor=None,
                 max_epochs: int = 50, label_smoothing: float = 0.0, gamma: float = 2.0):
        super().__init__()
        self.save_hyperparameters()

        self.feature_extractor_original = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        efficientnet_feature_dim = self.feature_extractor_original.num_features 

        self.feature_extractor_processed = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        
        combined_feature_dim = efficientnet_feature_dim * 2

        self.se_block = SEBlock(channel=combined_feature_dim)
        
        self.loss_weight_param = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        self.projection_head = nn.Sequential(
            nn.Linear(combined_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(combined_feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.supcon_loss_fn = EnhancedSupConLoss(
            temperature=supcon_temp,
            base_temperature=supcon_base_temp,
            contrast_mode='all',
            hard_mining_ratio=supcon_hard_mining_ratio, 
            margin=supcon_margin 
        )
        
        if class_weights_tensor is not None:
            alpha_for_focal = torch.tensor([class_weights_for_loss[0], class_weights_for_loss[1]], dtype=torch.float32)
        else:
            alpha_for_focal = None

        self.focal_loss_fn = FocalLoss(
            alpha=alpha_for_focal,
            gamma=self.hparams.gamma,
            label_smoothing=label_smoothing
        )
        
        self.class_weights_for_loss = class_weights_for_loss
        self.class_weights_tensor = class_weights_tensor

        self.train_raw_preds = []
        self.train_labels = []
        self.val_raw_preds = []
        self.val_labels = []
        self.test_raw_preds = []
        self.test_labels = []


    def forward(self, x):
        if isinstance(x, tuple) and len(x) == 4:
            img1_original, img2_original, img1_processed, img2_processed = x

            features1_original = self.feature_extractor_original(img1_original)
            features2_original = self.feature_extractor_original(img2_original)
            features1_processed = self.feature_extractor_processed(img1_processed)
            features2_processed = self.feature_extractor_processed(img2_processed)

            features1_combined = torch.cat((features1_original, features1_processed), dim=1)
            features2_combined = torch.cat((features2_original, features2_processed), dim=1)
            
            features1_combined_se = self.se_block(features1_combined)
            features2_combined_se = self.se_block(features2_combined)

            proj_features1 = self.projection_head(features1_combined_se)
            proj_features2 = self.projection_head(features2_combined_se)
            
            supcon_features = torch.stack((proj_features1, proj_features2), dim=1)
            
            logits = self.classification_head(features1_combined_se) 
            return supcon_features, logits
        elif isinstance(x, tuple) and len(x) == 2:
            img_original, img_processed = x
            features_original = self.feature_extractor_original(img_original)
            features_processed = self.feature_extractor_processed(img_processed)
            
            features_combined = torch.cat((features_original, features_processed), dim=1)
            
            features_combined_se = self.se_block(features_combined)
            
            logits = self.classification_head(features_combined_se)
            return logits
        else:
            raise ValueError("Unexpected input format for forward pass. Expected tuple of 2 or 4 tensors.")


    def training_step(self, batch, batch_idx):
        (imgs1_original, imgs2_original, imgs1_processed, imgs2_processed), labels = batch
        
        supcon_features, logits = self((imgs1_original, imgs2_original, imgs1_processed, imgs2_processed))

        supcon_loss = self.supcon_loss_fn(
            features=supcon_features,
            labels=labels,
            class_weights=self.class_weights_for_loss
        )

        ce_loss = self.focal_loss_fn(logits, labels)
        
        total_loss = self.loss_weight_param * supcon_loss + (1 - self.loss_weight_param) * ce_loss
        
        self.log('train_supcon_loss', supcon_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_ce_loss', ce_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('loss_weight_param', self.loss_weight_param, on_step=True, on_epoch=True, prog_bar=True)


        self.train_raw_preds.append(F.softmax(logits, dim=1).detach().cpu().numpy())
        self.train_labels.append(labels.cpu().numpy())
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        (imgs_original, imgs_processed), labels = batch
        logits = self((imgs_original, imgs_processed))
        
        loss = self.focal_loss_fn(logits, labels)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_raw_preds.append(F.softmax(logits, dim=1).detach().cpu().numpy())
        self.val_labels.append(labels.cpu().numpy())
        
        return loss

    def test_step(self, batch, batch_idx):
        (imgs_original, imgs_processed), labels = batch
        logits = self((imgs_original, imgs_processed))
        
        loss = self.focal_loss_fn(logits, labels)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        self.test_raw_preds.append(F.softmax(logits, dim=1).detach().cpu().numpy())
        self.test_labels.append(labels.cpu().numpy())
        
        return loss

    def on_train_epoch_end(self):
        if len(self.train_raw_preds) == 0:
            return
        
        all_raw_preds = np.concatenate(self.train_raw_preds)
        all_labels = np.concatenate(self.train_labels)
        
        all_preds_classes = np.argmax(all_raw_preds, axis=1)

        accuracy = accuracy_score(all_labels, all_preds_classes)
        self.log('train_accuracy_epoch', accuracy, prog_bar=True)

        f1 = f1_score(all_labels, all_preds_classes, average='weighted', zero_division=0)
        self.log('train_f1_epoch', f1, prog_bar=True)

        try:
            if len(np.unique(all_labels)) > 1:
                roc_auc = roc_auc_score(all_labels, all_raw_preds[:, 1], average='weighted')
                self.log('train_roc_auc_epoch', roc_auc, prog_bar=True)
            else:
                self.log('train_roc_auc_epoch', 0.0, prog_bar=True)
        except ValueError:
            self.log('train_roc_auc_epoch', 0.0, prog_bar=True)

        try:
            if len(np.unique(all_labels)) > 1:
                precision, recall, _ = precision_recall_curve(all_labels, all_raw_preds[:, 1])
                pr_auc = auc(recall, precision)
                self.log('train_pr_auc_epoch', pr_auc, prog_bar=True)
            else:
                self.log('train_pr_auc_epoch', 0.0, prog_bar=True)
        except ValueError:
            self.log('train_pr_auc_epoch', 0.0, prog_bar=True)
        
        self.train_raw_preds.clear()
        self.train_labels.clear()

    def on_validation_epoch_end(self):
        if len(self.val_raw_preds) == 0:
            return
        
        all_raw_preds = np.concatenate(self.val_raw_preds)
        all_labels = np.concatenate(self.val_labels)
        
        all_preds_classes = np.argmax(all_raw_preds, axis=1)

        accuracy = accuracy_score(all_labels, all_preds_classes)
        self.log('val_accuracy_epoch', accuracy, prog_bar=True)

        f1 = f1_score(all_labels, all_preds_classes, average='weighted', zero_division=0)
        self.log('val_f1_epoch', f1, prog_bar=True)

        try:
            if len(np.unique(all_labels)) > 1:
                roc_auc = roc_auc_score(all_labels, all_raw_preds[:, 1], average='weighted')
                self.log('val_roc_auc_epoch', roc_auc, prog_bar=True)
            else:
                self.log('val_roc_auc_epoch', 0.0, prog_bar=True)
        except ValueError:
            self.log('val_roc_auc_epoch', 0.0, prog_bar=True)

        try:
            if len(np.unique(all_labels)) > 1:
                precision, recall, _ = precision_recall_curve(all_labels, all_raw_preds[:, 1])
                pr_auc = auc(recall, precision)
                self.log('val_pr_auc_epoch', pr_auc, prog_bar=True)
            else:
                self.log('val_pr_auc_epoch', 0.0, prog_bar=True)
        except ValueError:
            self.log('val_pr_auc_epoch', 0.0, prog_bar=True)
        
        self.val_raw_preds.clear()
        self.val_labels.clear()

    def on_test_epoch_end(self):
        if len(self.test_raw_preds) == 0:
            print("No test predictions collected.")
            return
        
        all_raw_preds = np.concatenate(self.test_raw_preds)
        all_labels = np.concatenate(self.test_labels)
        
        all_preds_classes = np.argmax(all_raw_preds, axis=1)

        print("\n--- Comprehensive Evaluation Report (Test Set) ---")
        accuracy_test = accuracy_score(all_labels, all_preds_classes)
        print(f"Overall Accuracy (Test Set): {accuracy_test:.4f}")
        print("\nDetailed Classification Report (Test Set):")
        class_names = ['female', 'male'] 
        print(classification_report(all_labels, all_preds_classes, target_names=class_names, zero_division=0))

        print("\n--- Confusion Matrix (Test Set) ---")
        cm_test = confusion_matrix(all_labels, all_preds_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix (Test Set)')
        plt.show()

        try:
            if len(np.unique(all_labels)) > 1:
                roc_auc_test = roc_auc_score(all_labels, all_raw_preds[:, 1], average='weighted')
                print(f"Final Test ROC-AUC (weighted): {roc_auc_test:.4f}")
            else:
                print("Cannot compute ROC-AUC: Only one class present in test labels.")
        except Exception as e:
            print(f"Error computing final Test ROC-AUC: {e}")

        try:
            if len(np.unique(all_labels)) > 1:
                precision_test, recall_test, _ = precision_recall_curve(all_labels, all_raw_preds[:, 1])
                pr_auc_test = auc(recall_test, precision_test)
                print(f"Final Test PR-AUC (weighted): {pr_auc_test:.4f}")
            else:
                print("Cannot compute PR-AUC: Only one class present in test labels.")
        except Exception as e:
            print(f"Error computing final Test PR-auc: {e}")
        
        self.test_raw_preds.clear()
        self.test_labels.clear()

    def configure_optimizers(self):
        optimizer_params = [
            {'params': self.feature_extractor_original.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.feature_extractor_processed.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.se_block.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.projection_head.parameters(), 'lr': self.hparams.learning_rate * 2},
            {'params': self.classification_head.parameters(), 'lr': self.hparams.learning_rate * 2},
            {'params': self.loss_weight_param, 'lr': self.hparams.learning_rate * 5}
        ]

        optimizer = torch.optim.AdamW(optimizer_params, weight_decay=self.hparams.weight_decay)
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.hparams.max_epochs // 5,
                T_mult=2,
                eta_min=self.hparams.learning_rate / 100,
                verbose=True
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

# --- Utility Function for Visualizing Misclassifications ---
def visualize_misclassifications(model, dataloader, class_names, num_images=None):
    """
    Visualizes misclassified images with their true and predicted labels.
    If num_images is None, all misclassified images are shown.
    """
    misclassified_samples = []
    
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_idx, (imgs_tuple, labels) in enumerate(dataloader):
            imgs_original, imgs_processed = imgs_tuple
            
            imgs_original = imgs_original.to(device)
            imgs_processed = imgs_processed.to(device)
            labels = labels.to(device)

            logits = model((imgs_original, imgs_processed))
            predicted_probs = F.softmax(logits, dim=1)
            predicted_labels = torch.argmax(predicted_probs, dim=1)

            for i in range(len(labels)):
                if predicted_labels[i] != labels[i]:
                    img_cpu = imgs_original[i].cpu() 
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_display = img_cpu * std + mean
                    img_display = torch.clamp(img_display, 0, 1)
                    
                    misclassified_samples.append({
                        'image': img_display,
                        'true_label': class_names[labels[i].item()],
                        'predicted_label': class_names[predicted_labels[i].item()]
                    })
            if num_images is not None and len(misclassified_samples) >= num_images:
                break
    
    if misclassified_samples:
        display_count = len(misclassified_samples) if num_images is None else min(num_images, len(misclassified_samples))
        print(f"\n--- Visualizing {display_count} Misclassified Images ---")
        
        cols = 5
        rows = (display_count + cols - 1) // cols
        
        plt.figure(figsize=(15, 4 * rows))
        for i, sample in enumerate(misclassified_samples[:display_count]):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(sample['image'].permute(1, 2, 0).numpy())
            plt.title(f"True: {sample['true_label']}\nPred: {sample['predicted_label']}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo misclassified images found in the set provided for visualization!")
import collections
def run_task_b(TEST_PATH_TASK_B):
    CONFIG = {
        "BEST_MODEL_PATH": CHECKPOINT_PATH_TASK_B,
        "EMBEDDING_DIM": 512
    }
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {DEVICE} ---")

    class EmbeddingNet(nn.Module):
        def __init__(self, embedding_dim):
            super(EmbeddingNet, self).__init__()
            weights = EfficientNet_B4_Weights.DEFAULT
            self.backbone = efficientnet_b4(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Linear(in_features, embedding_dim)
            self.transforms = weights.transforms()

        def forward(self, x):
            return self.backbone(x)

    def prepare_evaluation_sets(data_path):
        reference_gallery_paths = {}
        query_set = []
        if not os.path.exists(data_path) or not os.path.isdir(data_path):
            print(f"Warning: Data path does not exist or is not a directory: {data_path}")
            return {}, [], []
        person_classes = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
        for class_name in person_classes:
            class_path = os.path.join(data_path, class_name)
            distortion_path = os.path.join(class_path, 'distortion')
            clean_images = [os.path.join(class_path, f) for f in os.listdir(class_path)
                            if f != 'distortion' and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if clean_images:
                reference_gallery_paths[class_name] = clean_images
            if os.path.exists(distortion_path):
                for img in os.listdir(distortion_path):
                    if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                        query_set.append((os.path.join(distortion_path, img), class_name))
        return reference_gallery_paths, query_set, person_classes

    def evaluate(model, ref_gallery_paths, query_set, person_classes, transform, device):
        print(f"\n--- Evaluating Model ---")
        model.to(device)
        model.eval()
        y_true, y_pred = [], []
        class_to_idx = {name: i for i, name in enumerate(person_classes)}
        if not query_set:
            print("No query images found. Check data structure.")
            return 0.0, 0.0
        print("Creating reference embedding gallery...")
        avg_reference_embeddings = collections.OrderedDict()
        with torch.no_grad():
            for class_name, img_paths in tqdm(ref_gallery_paths.items(), desc="Reference images"):
                embeddings = []
                for p in img_paths:
                    try:
                        img = Image.open(p).convert("RGB")
                        tensor = transform(img).unsqueeze(0).to(device)
                        embeddings.append(model(tensor))
                    except:
                        print(f"Error processing image {p}. Skipping.")
                        continue
                if embeddings:
                    avg_reference_embeddings[class_name] = torch.mean(torch.cat(embeddings), dim=0)
        if not avg_reference_embeddings:
            print("No reference embeddings found.")
            return 0.0, 0.0
        ref_labels = list(avg_reference_embeddings.keys())
        ref_embeds = torch.stack(list(avg_reference_embeddings.values()))
        print("Matching queries...")
        with torch.no_grad():
            for query_path, true_label in tqdm(query_set, desc="Query images"):
                try:
                    img_tensor = transform(Image.open(query_path).convert("RGB")).unsqueeze(0).to(device)
                    query_embedding = model(img_tensor)
                    q_norm = F.normalize(query_embedding)
                    r_norm = F.normalize(ref_embeds)
                    sims = torch.matmul(q_norm, r_norm.T)
                    best_match_idx = torch.argmax(sims, dim=1).item()
                    pred_label = ref_labels[best_match_idx]
                    y_true.append(class_to_idx[true_label])
                    y_pred.append(class_to_idx[pred_label])
                except:
                    continue
        if not y_true:
            print("No predictions made.")
            return 0.0, 0.0
        acc = (np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)) * 100
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        print(f"Accuracy: {acc:.2f}%, F1: {macro_f1:.4f}")
        return acc, macro_f1

    print("\n--- Running Task B Evaluation ---")
    NEW_TEST_FOLDER_PATH = TEST_PATH_TASK_B
    if not os.path.exists(NEW_TEST_FOLDER_PATH):
        print(f"Error: Folder not found → {NEW_TEST_FOLDER_PATH}")
        return
    ref_paths, query_set, classes = prepare_evaluation_sets(NEW_TEST_FOLDER_PATH)
    if not ref_paths or not query_set:
        print("Invalid test folder structure.")
        return
    model = EmbeddingNet(CONFIG["EMBEDDING_DIM"])
    try:
        model.load_state_dict(torch.load(CONFIG["BEST_MODEL_PATH"], map_location=DEVICE))
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    transform = EfficientNet_B4_Weights.DEFAULT.transforms()
    evaluate(model, ref_paths, query_set, classes, transform, DEVICE)

    
if __name__ == '__main__':
    print("Task A inference starts:\n")
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-5
    MAX_EPOCHS = 50 
    LABEL_SMOOTHING_EPSILON = 0.1
    FOCAL_LOSS_GAMMA = 3.0

    SUPCON_TEMP = 0.07 
    SUPCON_BASE_TEMP = 0.07 
    SUPCON_HARD_MINING_RATIO = 0.5
    SUPCON_MARGIN = 0.3

    VISUALIZE_ALL_MISCLASSIFICATIONS = True 

    # --- Step 1: Set up face detection model files (if not already present) ---
    print("--- Setting up face detection model files ---")
    setup_face_detection_models()

    # --- Step 2: Preprocess the Test Dataset ---
    print(f"\n--- Preprocessing Test Dataset from {TEST_PATH_TASK_A} to {TEST_PREPROCESSED_OUTPUT_DIR} ---")
    preprocess_and_detect_face(TEST_PATH_TASK_A, TEST_PREPROCESSED_OUTPUT_DIR)
    print("Test dataset preprocessing complete.")

    # --- Step 3: Initialize DataModule for Test Set ---
    data_module = GenderDataModule(
        train_data_dir=None,
        val_data_dir=None,
        train_preprocessed_data_dir=None,
        val_preprocessed_data_dir=None,
        test_data_dir=TEST_PATH_TASK_A,
        test_preprocessed_data_dir=TEST_PREPROCESSED_OUTPUT_DIR, # <--- MODIFIED HERE: Using the new explicit path
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    data_module.setup('test')

    if data_module.test_dataset is None or len(data_module.test_dataset) == 0:
        print("Error: Test dataset is empty or not loaded properly. Cannot proceed with evaluation.")
    else:
        # --- Step 4: Load the Trained Model from Checkpoint ---
        print(f"\n--- Loading model from checkpoint: {CHECKPOINT_PATH_TASK_A} ---")
        try:
            model = GenderClassificationModel.load_from_checkpoint(
                CHECKPOINT_PATH_TASK_A,
                map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                num_classes=2,
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                supcon_temp=SUPCON_TEMP,
                supcon_base_temp=SUPCON_BASE_TEMP,
                supcon_hard_mining_ratio=SUPCON_HARD_MINING_RATIO,
                supcon_margin=SUPCON_MARGIN,
                class_weights_for_loss=None,
                class_weights_tensor=None,
                max_epochs=MAX_EPOCHS,
                label_smoothing=LABEL_SMOOTHING_EPSILON,
                gamma=FOCAL_LOSS_GAMMA
            )
            model.eval()
            model.freeze()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            print("Model loaded successfully.")

            # --- Step 5: Perform Evaluation on Test Set ---
            print("\n--- Running Evaluation on Test Set ---")
            trainer = pl.Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                precision=16 if torch.cuda.is_available() else 32,
                logger=False
            )
            
            trainer.test(model, dataloaders=data_module.test_dataloader())

            # --- Step 6: Visualize Misclassifications on Test Set ---
            print("\n--- Visualizing Misclassifications on Test Set ---")
            class_names = list(data_module.test_dataset.idx_to_class.values())
            num_images_to_viz = None if VISUALIZE_ALL_MISCLASSIFICATIONS else 10
            visualize_misclassifications(
                model, 
                data_module.test_dataloader(), 
                class_names, 
                num_images=num_images_to_viz
            )

        except FileNotFoundError:
            print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH_TASK_A}. Please ensure the path is correct and the file exists.")
        except Exception as e:
            print(f"An error occurred during model loading or testing: {e}")
        print("\nTask A inference complete!\n")
    print("Task B evaluation starts:\n")
    run_task_b(TEST_PATH_TASK_B)
    print("\nTest pipeline execution complete!")