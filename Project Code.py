import cv2
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional
from PIL import Image

try:
    import mrcnn.config
    import mrcnn.model as modellib
    import mrcnn.utils
except ImportError:
    print("Error: Mask R-CNN library not found.  Please install it or add it to your PYTHONPATH.")
    print("  You may need to install the 'matterport' implementation: https://github.com/matterport/Mask_RCNN")

    class mrcnn:
        class config:
            pass
        class model:
            pass
        class utils:
            @staticmethod
            def extract_bboxes(masks):
                return np.array([])
            @staticmethod
            def compute_iou(box1, box2):
                return 0
            @staticmethod
            def mold_image(images, config):
                return images
            @staticmethod
            def unmold_detections(detections, mrcnn_mask, original_image_shape, image_shape, window, scale, config):
                return detections
        class visualize:
            @staticmethod
            def display_instances(image, boxes, masks, class_ids, class_names, scores=None):
                pass
        class Dataset:
            def load_image(self, image_id):
                return None
            def load_mask(self, image_id):
                return None, None
            def prepare(self, class_map=None):
                pass
            @property
            def image_ids(self):
                return []
            def image_reference(self, image_id):
                return ""
            def num_classes(self):
                return 0
            def class_names(self):
                return []
# Configuration Classes
class ClassificationConfig:
    """Configuration for the classification task."""
    IMAGE_SIZE = (224, 224)  # Default image size for EfficientNetB0
    BATCH_SIZE = 32
    EPOCHS = 10  # Reduced for initial testing
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 2 # set according to the dataset.
    TRAIN_DIR = "data/train"  #  Replace with your actual data paths
    VALID_DIR = "data/val"    #  Replace with your actual data paths
    TEST_DIR  = "data/test" # Replace
    MODEL_PATH = "classification_model.pth" # Path to save the classification model.  Use .pth for PyTorch
    CLASS_NAMES = ['class1', 'class2'] #Add more if needed

class SegmentationConfig(mrcnn.config.Config):
    """Configuration for the segmentation task (Mask R-CNN).
    Derives from the base Config class and overrides values specific
    to the toy dataset.
    """
    NAME = "custom"  # Override name
    # Train on 1 GPU and 1 images per GPU.  Batch size is 1 (per GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + your classes.  Set this!
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7 # Increased confidence threshold
    # Backbone network architecture
    # You can pick different backbones.  ResNet50 is a good balance.
    BACKBONE = "resnet50"
    # Use a smaller image size for faster training.  Set this as appropriate
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_DIM = 512
    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # Reduced anchor scales
    # Non-max suppression threshold to filter RPN proposals.
    RPN_NMS_THRESHOLD = 0.7
    # How many proposals to keep after non-max suppression
    PRE_NMS_TOP_COUNT = 1000
    POST_NMS_ROIS_TRAINING = 500
    POST_NMS_ROIS_INFERENCE = 1000
    # Reduce training ROIs per image to reduce memory load
    TRAIN_ROIS_PER_IMAGE = 128  # Reduced Train ROIs.
    # Use a small pool size to reduce the number of trainable parameters
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100 # Adjust based on dataset size
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7
    # Weight decay regularization
    WEIGHT_DECAY = 0.0001
    # Use a simple color scheme for visualization
    COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
              (0, 255, 255), (255, 0, 255), (255, 255, 0),
              (0, 0, 128), (0, 128, 0), (128, 0, 0),
              (0, 128, 128), (128, 0, 128), (128, 128, 0)]

def load_image(image_path: str) -> np.ndarray:
    """Loads an image from the given path.

    Args:
        image_path: Path to the image file.

    Returns:
        The loaded image as a NumPy array in RGB format.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resizes an image to the specified target size.

    Args:
        image: The image to resize as a NumPy array.
        target_size: The target size (width, height) as a tuple.

    Returns:
        The resized image as a NumPy array.
    """
    try:
        resized_image = cv2.resize(image, target_size)
        return resized_image
    except Exception as e:
        print(f"Error resizing image: {e}")
        return image  # Return original image on error to avoid crashing

def preprocess_image(image: np.ndarray, target_size: Tuple[int, int]) -> torch.Tensor:
    """Preprocesses an image for classification or other tasks.

    Args:
        image: The image to preprocess.
        target_size: The target size for resizing.

    Returns:
        The preprocessed image as a Torch Tensor.
    """
    image = resize_image(image, target_size)
    image = image.astype('float32') / 255.0  # Normalize pixel values
    image = torch.from_numpy(image).permute(2, 0, 1) # Convert to Tensor and change to (C, H, W)
    return image

def apply_mask(image: np.ndarray, mask: np.ndarray, color: List[int], alpha: float = 0.5) -> np.ndarray:
    """Applies a mask to an image.

    Args:
        image: The image to apply the mask to.
        mask: The mask to apply (a binary array).
        color: The color of the mask.
        alpha: Transparency of the mask.

    Returns:
        The image with the mask applied.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                 image[:, :, c] * (1 - alpha) + alpha * color[c],
                                 image[:, :, c])
    return image

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Computes the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1: First bounding box [x1, y1, x2, y2].
        box2: Second bounding box [x1, y1, x2, y2].

    Returns:
        The IoU value (float).
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0  # No intersection
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def compute_ap(gt_boxes: np.ndarray, gt_class_ids: np.ndarray,
               pred_boxes: np.ndarray, pred_class_ids: np.ndarray,
               pred_scores: np.ndarray, iou_threshold: float = 0.5) -> float:
    """Computes the Average Precision (AP) for a single class.

    Args:
        gt_boxes: Ground truth bounding boxes [N, (x1, y1, x2, y2)].
        gt_class_ids: Ground truth class IDs [N].
        pred_boxes: Predicted bounding boxes [M, (x1, y1, x2, y2)].
        pred_class_ids: Predicted class IDs [M].
        pred_scores: Predicted object scores [M].
        iou_threshold: IoU threshold to consider a detection a positive.

    Returns:
        The Average Precision (AP) for the given class.
    """
    # Sort predictions by score in descending order
    order = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[order]
    pred_class_ids = pred_class_ids[order]
    pred_scores = pred_scores[order]

    # Compute true positives and false positives
    num_gt = len(gt_boxes)
    true_positives = np.zeros((len(pred_boxes),))
    false_positives = np.zeros((len(pred_boxes),))
    
    for i, pred_box in enumerate(pred_boxes):
        matched_gt = -1
        for j, gt_box in enumerate(gt_boxes):
            # If class ID matches and IoU is above threshold, it's a match
            if gt_class_ids[j] == pred_class_ids[i] and compute_iou(gt_box, pred_box) >= iou_threshold:
                if matched_gt == -1:  # Only one match per prediction
                    true_positives[i] = 1
                    matched_gt = j
        if matched_gt == -1:
            false_positives[i] = 1

    # Compute precision and recall
    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)
    precision = cumulative_tp / (cumulative_tp + cumulative_fp)
    recall = cumulative_tp / num_gt

    # Compute AP as the area under the precision-recall curve
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0
    return ap

def compute_map(gt_boxes: List[np.ndarray], gt_class_ids: List[np.ndarray],
                pred_boxes: List[np.ndarray], pred_class_ids: List[np.ndarray],
                pred_scores: List[np.ndarray], iou_threshold: float = 0.5, num_classes: int = 2) -> float:
    """Computes the Mean Average Precision (mAP).

    Args:
        gt_boxes: List of ground truth bounding boxes.  Each element is [N, (x1, y1, x2, y2)].
        gt_class_ids: List of ground truth class IDs. Each element is [N].
        pred_boxes: List of predicted bounding boxes. Each element is [M, (x1, y1, x2, y2)].
        pred_class_ids: List of predicted class IDs. Each element is [M].
        pred_scores: List of predicted object scores.  Each element is [M].
        iou_threshold: IoU threshold.
        num_classes: Number of classes.

    Returns:
        The Mean Average Precision (mAP).
    """
    aps = []
    for class_id in range(1, num_classes):  # Exclude background class (0)
        # Extract ground truth and predictions for the current class
        class_gt_boxes = [box[gt_class_ids[i] == class_id]
                          for i, box in enumerate(gt_boxes) if any(gt_class_ids[i] == class_id)]
        class_pred_boxes = [box[pred_class_ids[i] == class_id]
                           for i, box in enumerate(pred_boxes) if any(pred_class_ids[i] == class_id)]
        class_pred_class_ids = [class_id * np.ones_like(pred_class_ids[i][pred_class_ids[i] == class_id])
                                for i in range(len(pred_class_ids)) if any(pred_class_ids[i] == class_id)]
        class_pred_scores = [score[pred_class_ids[i] == class_id]
                               for i, score in enumerate(pred_scores) if any(pred_class_ids[i] == class_id)]

        # If there are no GT boxes for this class, return 0
        if len(class_gt_boxes) == 0:
            aps.append(0)
            continue
        
        ap = compute_ap(np.concatenate(class_gt_boxes, axis=0) if class_gt_boxes else np.zeros((0,4)),
                          np.concatenate([np.array([class_id] * len(x)) for x in class_gt_boxes], axis=0) if class_gt_boxes else np.zeros((0,)),
                          np.concatenate(class_pred_boxes, axis=0) if class_pred_boxes else np.zeros((0,4)),
                          np.concatenate(class_pred_class_ids, axis=0) if class_pred_class_ids else np.zeros((0,)),
                          np.concatenate(class_pred_scores, axis=0) if class_pred_scores else np.zeros((0,)),
                          iou_threshold)
        aps.append(ap)
    return np.mean(aps)

# Classification section
class CustomDataset(Dataset):
    """Custom dataset class for loading and preprocessing images for classification."""
    def __init__(self, data_dir: str, transform: transforms.Compose):
        """
        Args:
            data_dir: Directory containing the image data.
            transform: Transformations to apply to the images.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))
        self.image_paths = []
        self.labels = []
        for i, cls_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(class_dir):
                continue  # Skip if it's not a directory
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(i)

    def __len__(self) -> int:
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Gets the image and its corresponding label at the given index.

        Args:
            idx: Index of the image to retrieve.

        Returns:
            A tuple containing the image (as a PyTorch Tensor) and its label (as an integer).
        """
        img_path = self.image_paths[idx]
        image = load_image(img_path) # Use the load_image function defined earlier
        image = Image.fromarray(image) # Convert numpy array to PIL Image
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

def create_classification_model(num_classes: int) -> nn.Module:
    """Creates a classification model based on EfficientNetB0.

    Args:
        num_classes: The number of classes for the classification task.

    Returns:
        A PyTorch model (nn.Module) for image classification.
    """
    # Load pre-trained EfficientNetB0
    efficientnet = models.efficientnet_b0(pretrained=True)

    # Freeze all the parameters in the feature extractor
    for param in efficientnet.parameters():
        param.requires_grad = False

    # Replace the classification head with a custom one
    in_features = efficientnet.classifier[1].in_features
    efficientnet.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes),
    )
    return efficientnet

def train_classification_model(model: nn.Module,
                             train_loader: DataLoader,
                             val_loader: DataLoader,
                             criterion: nn.Module,
                             optimizer: optim.Optimizer,
                             config: ClassificationConfig,
                             device: torch.device) -> None:
    """Trains the classification model.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        criterion: The loss function.
        optimizer: The optimizer.
        config: The classification configuration.
        device: The device to train on (CPU or GPU).
    """
    best_val_loss = float('inf')
    for epoch in range(config.EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss = train_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss = val_loss / len(val_loader.dataset)

        # Print epoch statistics
        print(f"Epoch {epoch+1}/{config.EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Calculate and print classification report and confusion matrix
        report = classification_report(all_labels, all_preds, target_names=config.CLASS_NAMES)
        matrix = confusion_matrix(all_labels, all_preds)
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", matrix)

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"Saved model to {config.MODEL_PATH}")
            
def test_classification_model(model: nn.Module, test_loader: DataLoader, config: ClassificationConfig, device: torch.device) -> None:
    """Tests the classification model and prints metrics.

    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for the test set.
        config: The classification configuration.
        device: The device to run the testing on (CPU or GPU).
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss() #define criterion
    with torch.no_grad():  # Disable gradient calculation during testing
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)  # Get the predicted class indices
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}")

    # Generate and print the classification report
    report = classification_report(all_labels, all_preds, target_names=config.CLASS_NAMES)
    print("Classification Report (Test Set):\n", report)

    # Generate and print the confusion matrix
    matrix = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix (Test Set):\n", matrix)
    
# Segmentation Section
class CustomSegmentationDataset(mrcnn.Dataset):
    """Custom dataset class for loading and preprocessing data for segmentation with Mask R-CNN."""
    def __init__(self, image_dir: str, class_names: List[str]):
        """
        Args:
            image_dir: Directory containing the images and masks.
            class_names: List of class names (including background).
        """
        super().__init__()
        self.image_dir = image_dir
        self.class_names = class_names
        self.class_ids_map = {class_name: i for i, class_name in enumerate(class_names)}

    def load_dataset(self, image_dir: str):
        """Load a subset of the dataset.
        Args:
            image_dir: Root directory of the dataset.
        """
        # Add classes.
        for i, class_name in enumerate(self.class_names):
            self.add_class("custom", i, class_name)

        # List all images in the directory.  Assumes masks are in a subdirectory
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_id = filename.split('.')[0]  # Remove extension
                image_path = os.path.join(image_dir, filename)
                mask_path = os.path.join(image_dir, "masks", image_id + ".png") #mask should be png
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask file not found: {mask_path} for image {image_path}. Skipping.")
                    continue
                self.add_image(
                    "custom",
                    image_id=image_id,
                    path=image_path,
                    mask_path=mask_path)

    def load_image(self, image_id: int) -> np.ndarray:
        """Load the image associated with the given image ID.
        Args:
            image_id: Theid of the image to load
        Returns:
            (ndarray) Image as a numpy array
        """
        image_info = self.image_info[image_id]
        image_path = image_info['path']
        return load_image(image_path)  # Use the load_image function

    def load_mask(self, image_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load the masks and class IDs for the given image ID.

           Function heavily adapted from the matterport Mask-RCNN implementation.
        Args:
            image_id: The id of the image to load masks for
        Returns:
            (ndarray) A boolean mask array of shape [height, width, instance count]
            (ndarray) class ID array [instance count]
        """
        image_info = self.image_info[image_id]
        mask_path = image_info['mask_path']

        # Read the mask image.  This should be a single-channel image where
        # each color represents a different instance.
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
           raise Exception(f"Failed to read mask image at {mask_path}")
        # Get the height and width of the mask.
        height, width = mask.shape[:2]
        # Find unique colors (instance IDs) in the mask.  Exclude the background color (0).
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids != 0]
        num_instances = len(instance_ids)
        if num_instances == 0:
            # If there are no instances, return an empty mask and an empty class ID array.
            return np.zeros((height, width, 0), dtype=np.bool_), np.zeros((0,), dtype=np.int32)

        # Create an empty mask array.  Initialize it as a boolean array.
        masks = np.zeros((height, width, num_instances), dtype=np.bool_)
        class_ids = np.zeros((num_instances,), dtype=np.int32)

        # Loop over the unique instance IDs (colors) in the mask image.
        for i, instance_id in enumerate(instance_ids):
            # Create a binary mask for the current instance ID.
            instance_mask = (mask == instance_id)
            masks[:, :, i] = instance_mask
            # Get the class ID for the current instance ID.  Assume the instance ID
            # corresponds to the class ID.  You might need to adjust this mapping
            # based on how your mask images are encoded.
            class_ids[i] = 1  #  Assume only one class for now.  Change this as needed.

        return masks.astype(np.bool_), class_ids.astype(np.int32)
    
def load_mrcnn_model(config: SegmentationConfig, model_path: str) -> modellib.MaskRCNN:
    """Loads the Mask R-CNN model from the specified path.

    Args:
        config: The segmentation configuration.
        model_path: Path to the Mask R-CNN model file.

    Returns:
        The loaded Mask R-CNN model.
    """
    # Create model object.
    model = modellib.MaskRCNN(mode="inference", model_dir="./", config=config)

    # Load weights trained on MS-COCO
    print(f"Loading weights from {model_path}")
    model.load_weights(model_path, by_name=True)
    return model

def detect_instances(image: np.ndarray, model: modellib.MaskRCNN, config: SegmentationConfig) -> List[dict]:
    """Detects object instances in the given image using the Mask R-CNN model.

    Args:
        image: The image to detect objects in.
        model: The Mask R-CNN model.
        config: The segmentation configuration.

    Returns:
        A list of dictionaries, where each dictionary represents a detected object
        instance and contains keys like 'rois', 'masks', 'class_ids', 'scores'.
    """
    # Run detection
    results = model.detect([image], verbose=0)
    return results[0]  # Return the first image's results

def visualize_segmentation(image: np.ndarray, results: dict, config: SegmentationConfig, class_names: List[str]) -> None:
    """Visualizes the segmentation results on the image.

    Args:
        image: The original image.
        results: The detection results dictionary.
        config: The segmentation configuration.
        class_names: List of class names.
    """
    boxes = results['rois']
    masks = results['masks']
    class_ids = results['class_ids']
    scores = results['scores']

    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("No instances to display")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0] == scores.shape[0]

    for i in range(N):
        color = config.COLORS[i % len(config.COLORS)]
        # Bounding box
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Label and score
        class_name = class_names[class_ids[i]]
        score = scores[i] if scores is not None else None
        label = "{}{:d} {:.3f}".format("", class_ids[i], score) if score else "{}{:d}".format("", class_ids[i])
        cv2.putText(image, label, (x1, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Mask
        mask = masks[:, :, i]
        image = apply_mask(image, mask, color)
    cv2.imshow('Segmented Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def evaluate_segmentation(gt_boxes: List[np.ndarray], gt_class_ids: List[np.ndarray],
                          pred_boxes: List[np.ndarray], pred_class_ids: List[np.ndarray],
                          pred_scores: List[np.ndarray], config: SegmentationConfig) -> float:
    """Evaluates the segmentation results using Mean Average Precision (mAP).

    Args:
        gt_boxes: List of ground truth bounding boxes [N, (x1, y1, x2, y2)].
        gt_class_ids: List of ground truth class IDs [N].
        pred_boxes: List of predicted bounding boxes [M, (x1, y1, x2, y2)].
        pred_class_ids: List of predicted class IDs [M].
        pred_scores: List of predicted object scores [M].
        config: The segmentation configuration.

    Returns:
        The Mean Average Precision (mAP).
    """
    return compute_map(gt_boxes, gt_class_ids, pred_boxes, pred_class_ids, pred_scores, iou_threshold=0.5, num_classes=config.NUM_CLASSES)

def main():
    """Main function to run the image processing, classification, and segmentation pipeline."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Configuration
    classification_config = ClassificationConfig()
    segmentation_config = SegmentationConfig()

    # Device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Classification
    print("--- Classification ---")
    # Define transformations for the training, validation, and test sets
    train_transforms = transforms.Compose([
        transforms.Resize(classification_config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(classification_config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(classification_config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create the datasets
    train_dataset = CustomDataset(classification_config.TRAIN_DIR, train_transforms)
    val_dataset = CustomDataset(classification_config.VALID_DIR, val_transforms)
    test_dataset = CustomDataset(classification_config.TEST_DIR, test_transforms)

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=classification_config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=classification_config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=classification_config.BATCH_SIZE, shuffle=False)

    # Create the model
    classification_model = create_classification_model(classification_config.NUM_CLASSES).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classification_model.parameters(), lr=classification_config.LEARNING_RATE)

    # Train the model
    train_classification_model(classification_model, train_loader, val_loader, criterion, optimizer, classification_config, device)
    
    # Load the best model
    classification_model.load_state_dict(torch.load(classification_config.MODEL_PATH))

    # Test the model
    test_classification_model(classification_model, test_loader, classification_config, device)

    # 2. Segmentation
    print("\n--- Segmentation ---")
    # Load Mask R-CNN model (assuming it's pre-trained)
    # Replace 'path/to/your/mask_rcnn_model.h5' with the actual path to your pre-trained Mask R-CNN model
    mrcnn_model_path = 'mask_rcnn_model.h5' # IMPORTANT: Replace with your model path
    segmentation_model = load_mrcnn_model(segmentation_config, mrcnn_model_path)

    # Load and prepare the custom dataset.  Use the class defined earlier.
    train_seg_dataset = CustomSegmentationDataset(classification_config.TRAIN_DIR, class_names=['BG','class1']) # Provide class names
    train_seg_dataset.load_dataset(classification_config.TRAIN_DIR)
    train_seg_dataset.prepare()

    # Example of how to use the segmentation model (inference on a single image)
    image_path = os.path.join(classification_config.TEST_DIR, test_dataset.image_paths[0].split('/')[-1])  # Path to an example image
    image = load_image(image_path)
    results = detect_instances(image, segmentation_model, segmentation_config)
    print("Detection results:", results)
    visualize_segmentation(image.copy(), results, segmentation_config, class_names=['BG','class1'])  # Visualize the results
    
    # Example Evaluation on a few images.
    gt_boxes = []
    gt_class_ids = []
    pred_boxes = []
    pred_class_ids = []
    pred_scores = []
    image_ids = train_seg_dataset.image_ids[:5] # evaluate on first 5 images.

    for image_id in image_ids:
        # Load ground truth data using the dataset's load_mask method
        gt_masks, gt_classes = train_seg_dataset.load_mask(image_id)
        gt_bbox = mrcnn.utils.extract_bboxes(gt_masks)  # Use the utils.extract_bboxes
        gt_boxes.append(gt_bbox)
        gt_class_ids.append(gt_classes)

        # Load image and make prediction
        image = train_seg_dataset.load_image(image_id)
        results = detect_instances(image, segmentation_model, segmentation_config)

        # Extract prediction data
        pred_boxes.append(results['rois'])
        pred_class_ids.append(results['class_ids'])
        pred_scores.append(results['scores'])

    # Compute mAP
    map_value = evaluate_segmentation(gt_boxes, gt_class_ids, pred_boxes, pred_class_ids, pred_scores, segmentation_config)
    print(f"Mean Average Precision (mAP): {map_value:.4f}")

if __name__ == "__main__":
    main()
