import os
from pathlib import Path
import logging

import hydra
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Imports that require root directory setup
from src.models.dogbreed_classifier import DogBreedClassifier
from src.utils.logging_utils import setup_logger, task_wrapper, get_rich_progress

log = logging.getLogger(__name__)

# Add this near the beginning of your infer.py script
CLASS_NAMES = ["Beagle", "Boxer", "Bulldog", "Dachshund", "German Shepherd", 
               "Golden Retriever", "Labrador Retriever", "Poodle", "Rottweiler", 
               "Yorkshire Terrier"]

@task_wrapper
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return img, transform(img).unsqueeze(0)

@task_wrapper
def infer(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # Use the manually defined CLASS_NAMES if not present in model.hparams
    class_labels = getattr(model.hparams, 'class_names', CLASS_NAMES)
    predicted_label = class_labels[predicted_class]
    confidence = probabilities[0][predicted_class].item()
    return predicted_label, confidence

@task_wrapper
def save_prediction_image(image, predicted_label, confidence, output_path):
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_label} (Confidence: {confidence:.2f})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

@task_wrapper
def process_images(cfg: DictConfig):
    model = DogBreedClassifier.load_from_checkpoint(cfg.ckpt_path)
    if not hasattr(model.hparams, 'class_names'):
        model.hparams.class_names = CLASS_NAMES
    model.eval()
    input_folder = Path(cfg.input_folder)
    output_folder = Path(cfg.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    image_files = list(input_folder.glob('*'))
    print(image_files)
    with get_rich_progress() as progress:
        task = progress.add_task("[green]Processing images...", total=len(image_files))
        
        for image_file in image_files:
            if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img, img_tensor = load_image(image_file)
                predicted_label, confidence = infer(model, img_tensor.to(model.device))
                
                output_file = output_folder / f"{image_file.stem}_prediction.png"
                save_prediction_image(img, predicted_label, confidence, output_file)
                
                progress.console.print(f"Processed {image_file.name}: {predicted_label} ({confidence:.2f})")
                progress.advance(task)

@hydra.main(version_base=None, config_path="../configs", config_name="infer")
def main(cfg: DictConfig):
    # Set up paths
    log_dir = Path(cfg.paths.log_dir)
    checkpoint_dir = Path(cfg.paths.checkpoint_dir)

    # Set up logger
    setup_logger(log_dir / "infer_log.log")

    # Find the latest checkpoint
    checkpoints = list(checkpoint_dir.glob('*.ckpt'))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    latest_checkpoint = max(checkpoints, key=os.path.getctime)

    # Update the checkpoint path in the config
    cfg.ckpt_path = str(latest_checkpoint)

    log.info(f"Using checkpoint: {cfg.ckpt_path}")

    # Process images
    process_images(cfg)

if __name__ == "__main__":
    main()