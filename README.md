# Vision-Transformer
Vision Transformer vs Convolution Neural Network

1. Open the ".py" file
2. Download the Data set from this link : https://huggingface.co/datasets/student/CIFAR-10/tree/main
3. Adjust the address of train and test directories
4. Run the code

# CIFAR-10 Image Classification with Vision Transformer

## Description
This project implements a *Vision Transformer (ViT) model for image classification on a custom CIFAR-10 dataset. The dataset consists of 32x32 RGB images across 10 classes (plane, car, bird, cat, deer, dog, frog, horse, ship, truck). The model uses a pre-trained `vit_tiny_patch16_224` from the `timm` library, fine-tuned for CIFAR-10. The project includes data loading, training with mixed precision, evaluation, and single-image prediction capabilities.

## Features
- Dataset: Custom CIFAR-10 dataset with PNG images organized in train/test folders.
- Model: Pre-trained Vision Transformer (`vit_tiny_patch16_224`) fine-tuned for 10 classes.
- Training: Uses AdamW optimizer, cross-entropy loss, StepLR scheduler, and mixed precision for efficiency.
- Evaluation: Computes test accuracy.
- Prediction: Supports single-image classification with class name output.
- Error Handling*: Skips invalid images during dataset loading.

## Prerequisites
- Python 3.8+
- Libraries:
  - `torch`
  - `torchvision`
  - `timm`
  - `Pillow`
  - `tqdm`
- CUDA-enabled GPU (optional, for faster training)
- Custom CIFAR-10 dataset (PNG images in train/test folders)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cifar10-vit-classification.git
   cd cifar10-vit-classification

#Install dependencies:
    pip install torch torchvision timm Pillow tqdm

#Dataset Setup:
Prepare the custom CIFAR-10 dataset (not included in the repository due to size).
Place images in two folders: cifar-train and cifar-test, with filenames like plane_001.png, car_002.png, etc.

#Example structure:
cifar-train/
├── plane_001.png
├── car_002.png
├── bird_003.png
└── ...
cifar-test/
├── plane_101.png
├── dog_102.png
└── ...
Update train_extracted_path and test_extracted_path in the script if your paths differ (default: C:/Users/mersad/Desktop/cifar-train and cifar-test).

##Usage
#Prepare the Dataset:
Ensure the train and test folders contain PNG images with class names in filenames (e.g., cat_001.png).
Verify paths in the script match your dataset location.

#Run the Script:
python cifar10_vit.py

#This will:
Load and preprocess images (resize to 224x224, normalize).
Train the ViT model for 5 epochs with mixed precision.
Evaluate on the test set and print accuracy.
Save the trained model as vit_cifar10.pth.
Predict the class of a single example image (default: C:/Users/mersad/Desktop/image.png).

#Single Image Prediction:
To classify a new image, update single_image_path in the script or call:
    predicted_label = predict_single_image("path/to/your/image.png", model, transform, device)
    print(f"Predicted label: {predicted_label}")

#Code Structure
Dataset: Custom CIFAR10CustomDataset loads PNG images, maps class names to labels (0-9), and skips invalid files.
Preprocessing: Resizes images to 224x224, normalizes using ImageNet means/stds.
Model: ViT (vit_tiny_patch16_224) with a modified head for 10 classes.
Training: Uses AdamW (lr=3e-4), StepLR scheduler (step=7, gamma=0.1), and mixed precision via torch.cuda.amp.
Evaluation: Calculates test accuracy.
Prediction: Predicts class names for single images using reverse label mapping.

#Results
Test Accuracy: Printed after evaluation (e.g., Test Accuracy: 92.50%).
Single Image Prediction: Outputs the class name (e.g., The predicted label for the image is: dog).

#Example
For an image dog_101.png:
Predicted label: dog.

#Notes
Dataset: Excluded from the repository (see .gitignore). Users must provide their own CIFAR-10 dataset in the specified format.
Paths: Update train_extracted_path, test_extracted_path, and single_image_path to match your local setup.
Performance: The model is trained for 5 epochs. Increase epochs or tune lr for better accuracy.
Hardware: Mixed precision reduces GPU memory usage, but CPU training is supported.
Image Format: Only PNG images are processed. Ensure filenames start with
