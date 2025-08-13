# Autoencoder for Image Compression and Inpainting
This repository includes code for implementing an autoencoder to perform either image compression or image inpainting, using a feed-forward neural network or convolutional neural network architecture. It was created as a personal project to develop practical experience with CNNs, encoder-decoder architectures, and training workflows in PyTorch. 

## How to use this code
This project uses the dog class of images in the CIFAR-10 image dataset. 

The basic command to train image compression: 
```
python train_compression.py
```
Runs with default settings:
```
--model-name CNN 
--epochs 101
--lr 1e-3
--batch-size 32
--dog-class-index 5
--seed 0
--epochs-per-summary 10
```
Auto-detects device (cuda if available, else cpu)

```--model-name``` accepts only CNN or NN.

#### Example (all modification options)
```
python train_compression.py \
  --model-name CNN \
  --epochs 50 \
  --lr 5e4 \
  --batch-size 64 \
  --dog-class-index 5 \
  --device cuda \
  --seed 123 \
  --checkpoint-dir checkpoints \
  --epochs-per-summary 100 \
  --restore-path checkpoints/model_010.pth
```
## Files
```models.py``` | Creates classes for the two autoencoder models, "NN_Autoencoder" (feed-forward) and "CNN_Autoencoder" (convolutional).

```utils.py``` | Includes helper functions (e.g. to mask images for inpainting, to visualize outputs).

```train_compression.py``` | Training script for image compression.

```train_inpainting.py``` | Training script for image inpainting.

## Example output
<img width="400" height="282" alt="Screen Shot 2025-08-12 at 3 30 56 PM" src="https://github.com/user-attachments/assets/7baf2126-08f7-470b-bb21-efe452ef97af" />
<img width="408" height="285" alt="Screen Shot 2025-08-12 at 3 39 20 PM" src="https://github.com/user-attachments/assets/fa11a10f-831a-49f7-8585-3d0040d38ce6" />
<img width="400" height="373" alt="Screen Shot 2025-08-13 at 3 01 21 PM" src="https://github.com/user-attachments/assets/d323f5d8-9c4f-4c3b-99dd-98ac6ec5d9a9" />

 
