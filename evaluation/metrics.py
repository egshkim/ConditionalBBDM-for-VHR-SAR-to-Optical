import os
import torch
import torchvision.transforms as transforms
from torchmetrics.image import (
    SpectralAngleMapper, 
    ErrorRelativeGlobalDimensionlessSynthesis,
    SpectralDistortionIndex,  # The one available in torchmetrics
)
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime
from fsim import FSIM, FSIMc
from CW_SSIM import CW_SSIM

def load_image(image_path):
    """Load an image and convert it to a PyTorch tensor with shape (C, H, W)."""
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    tensor_img = transform(img)
    tensor_img = torch.clamp(tensor_img, min=1e-5, max=1.0)
    return tensor_img

def calculate_spatial_distortion(pred, target, window_size=7):
    """Calculate spatial distortion between two images."""
    # Add small epsilon to avoid division by zero
    eps = 1e-5
    
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    if torch.cuda.is_available():
        kernel_x = kernel_x.cuda()
        kernel_y = kernel_y.cuda()
        pred = pred.cuda()
        target = target.cuda()

    ds_channels = []
    
    for c in range(pred.shape[1]):
        pred_channel = pred[:, c:c+1, :, :]
        target_channel = target[:, c:c+1, :, :]
        
        grad_pred_x = torch.nn.functional.conv2d(pred_channel, kernel_x, padding=1)
        grad_pred_y = torch.nn.functional.conv2d(pred_channel, kernel_y, padding=1)
        grad_target_x = torch.nn.functional.conv2d(target_channel, kernel_x, padding=1)
        grad_target_y = torch.nn.functional.conv2d(target_channel, kernel_y, padding=1)

        grad_pred = torch.sqrt(grad_pred_x**2 + grad_pred_y**2 + eps)
        grad_target = torch.sqrt(grad_target_x**2 + grad_target_y**2 + eps)

        ds_channel = torch.mean(torch.abs(grad_pred - grad_target))
        ds_channels.append(ds_channel.item())

    return np.mean(ds_channels)

def calculate_qnr(pred, target, alpha=1, beta=1):
    """Calculate Quality with No Reference (QNR) metric."""
    eps = 1e-8
    try:
        # Calculate spectral distortion
        spectral_metric = SpectralDistortionIndex()
        d_lambda = spectral_metric(pred, target).item()
        
        # Ensure values are within valid range [0,1]
        d_lambda = np.clip(d_lambda, 0, 1)
        
        # Calculate spatial distortion
        d_s = calculate_spatial_distortion(pred, target)
        d_s = np.clip(d_s, 0, 1)
        
        # Calculate QNR with protection against invalid values
        qnr = ((1 - d_lambda + eps)**alpha) * ((1 - d_s + eps)**beta)
        
        return np.clip(qnr, 0, 1)  # Ensure output is in valid range
    except Exception as e:
        print(f"Error in QNR calculation: {str(e)}")
        return None

def compute_metrics_for_folders(synthesized_folder, ground_truth_folder, ratio=4, save_log=True):
    """Compute all metrics between paired images from two folders."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Initialize metrics and move them to device
    sam_metric = SpectralAngleMapper(reduction="elementwise_mean").to(device)
    ergas_metric = ErrorRelativeGlobalDimensionlessSynthesis(ratio=ratio, reduction="elementwise_mean").to(device)
    spectral_metric = SpectralDistortionIndex().to(device)
    
    # Initialize FSIM and CW-SSIM metrics
    fsim_metric = FSIMc().to(device)
    cwssim_metric = CW_SSIM(imgSize=[512,512], channels=3, level=4, ori=8, device=device).to(device)
    
    print("\nInitialized metrics on device:", device)
    
    # Debug CW-SSIM components specifically
    print("\nCW-SSIM component devices:")
    if hasattr(cwssim_metric, 'win7'):
        print(f"win7 device: {cwssim_metric.win7.device}")
    if hasattr(cwssim_metric, 'w'):
        print(f"w device: {cwssim_metric.w.device}")
    if hasattr(cwssim_metric, 'SP'):
        print("SP components:")
        if hasattr(cwssim_metric.SP, 'hl0'):
            print(f"  hl0 device: {cwssim_metric.SP.hl0.device}")
        if hasattr(cwssim_metric.SP, 'indF'):
            for i, component in enumerate(cwssim_metric.SP.indF):
                print(f"  indF[{i}] device: {component.device}")
        if hasattr(cwssim_metric.SP, 'indB'):
            for i, component in enumerate(cwssim_metric.SP.indB):
                print(f"  indB[{i}] device: {component.device}")
    
    
    synthesized_images = sorted(os.listdir(synthesized_folder))
    ground_truth_images = sorted(os.listdir(ground_truth_folder))
    
    sam_values = []
    ergas_values = []
    spectral_dist_values = []
    spatial_dist_values = []
    qnr_values = []
    chd_values = []
    fsim_values = []
    cwssim_values = []
    
    for filename in tqdm(synthesized_images, desc="Computing Metrics"):
        if filename in ground_truth_images:
            try:
                synth_img_path = os.path.join(synthesized_folder, filename)
                gt_img_path = os.path.join(ground_truth_folder, filename)
                
                synth_img = load_image(synth_img_path).to(device)
                gt_img = load_image(gt_img_path).to(device)
                
                print(f"\nProcessing {filename}")
                print(f"Synthesized image device: {synth_img.device}")
                print(f"Ground truth image device: {gt_img.device}")
                
                if synth_img.shape != gt_img.shape:
                    print(f"Skipping {filename}: Image shapes do not match")
                    continue
                
                # Add more aggressive clamping here
                eps = 1e-5  # Larger epsilon
                synth_img = torch.clamp(synth_img, min=eps, max=1.0)
                gt_img = torch.clamp(gt_img, min=eps, max=1.0)
                # print(f"After clamping - Pred range: [{synth_img.min()}, {synth_img.max()}]")
                # print(f"After clamping - Target range: [{gt_img.min()}, {gt_img.max()}]")
                # Check means
                pred_means = synth_img.mean(dim=(1,2))
                target_means = gt_img.mean(dim=(1,2))
                # print(f"Pred channel means: {pred_means}")
                # print(f"Target channel means: {target_means}")
                
                # Check for invalid values before computation
                if torch.isnan(synth_img).any() or torch.isinf(synth_img).any() or \
                   torch.isnan(gt_img).any() or torch.isinf(gt_img).any():
                    print(f"Found invalid values in {filename}")
                    continue
                
                synth_img = synth_img.unsqueeze(0)
                gt_img = gt_img.unsqueeze(0)
                
                try:
                    # Debug inside spectral distortion calculation
                    print("\nCalculating Spectral Distortion:")
                    spectral_dist = spectral_metric(synth_img, gt_img)
                    print(f"Raw spectral distortion value: {spectral_dist}")
                    
                    if torch.isnan(spectral_dist) or torch.isinf(spectral_dist):
                        print("WARNING: Invalid spectral distortion value detected")
                        print("Detailed tensor stats:")
                        print(f"Pred stats - min: {synth_img.min()}, max: {synth_img.max()}, mean: {synth_img.mean()}")
                        print(f"Target stats - min: {gt_img.min()}, max: {gt_img.max()}, mean: {gt_img.mean()}")
                        continue
                        
                    spectral_dist = spectral_dist.item()
                    spectral_dist_values.append(spectral_dist)
                    print(f"Final spectral distortion value: {spectral_dist}")
                    
                except Exception as e:
                    print(f"Error in spectral distortion calculation: {str(e)}")
                    print("Tensor shapes:")
                    print(f"Pred shape: {synth_img.shape}")
                    print(f"Target shape: {gt_img.shape}")
                    continue
                
                # Calculate FSIM (expects input range 0-255)
                fsim_score = fsim_metric(synth_img * 255, gt_img * 255)
                if not torch.isnan(fsim_score):
                    fsim_values.append(fsim_score.item())
                
                # Try CW-SSIM computation with detailed logging
                try:
                    print("\nStarting CW-SSIM computation:")
                    print(f"Input tensor devices - Synth: {synth_img.device}, GT: {gt_img.device}")
                    
                    # Add debug prints inside CW-SSIM forward pass
                    cwssim_score = cwssim_metric(synth_img, gt_img, as_loss=False, debug=True)
                    print(f"CW-SSIM computation successful, score: {cwssim_score.item()}")
                    
                    if not torch.isnan(cwssim_score):
                        cwssim_values.append(cwssim_score.item())
                except Exception as e:
                    print(f"CW-SSIM computation failed: {str(e)}")
                    print(f"Exception type: {type(e)}")
                    import traceback
                    print(traceback.format_exc())
                    
                # Calculate metrics with value checking
                sam_score = sam_metric(synth_img, gt_img).item()
                if not np.isnan(sam_score):
                    sam_values.append(sam_score)

                ergas_score = ergas_metric(synth_img, gt_img).item()
                if not np.isnan(ergas_score):
                    ergas_values.append(ergas_score)
                
                
                
                # spatial_dist = calculate_spatial_distortion(synth_img, gt_img)
                # if not np.isnan(spatial_dist):
                #     spatial_dist_values.append(spatial_dist)
                
                # qnr_score = calculate_qnr(synth_img, gt_img)
                # if qnr_score is not None and not np.isnan(qnr_score):
                #     qnr_values.append(qnr_score)
                
                
                try:
                    # Before computing CHD, move tensors to CPU
                    synth_img_cpu = synth_img.cpu()
                    gt_img_cpu = gt_img.cpu()
                    chd_score = compute_chd(synth_img_cpu, gt_img_cpu)
                    if not np.isnan(chd_score):
                        chd_values.append(chd_score)
                except Exception as e:
                    if filename == synthesized_images[0]:  # Only print for first file
                        print(f"CHD computation failed: {str(e)}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                print(f"Exception type: {type(e)}")
                import traceback
                print(traceback.format_exc())
                continue
    
    # Create results dictionary with metadata
    final_results = {
        'metadata': {
            'synthesized_folder': synthesized_folder,
            'ground_truth_folder': ground_truth_folder,
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            'ratio': ratio,
            'num_images_processed': len(synthesized_images),
        },
        'metrics': {
            'SAM': np.mean(sam_values) if sam_values else None,
            'ERGAS': np.mean(ergas_values) if ergas_values else None,
            'Spectral_Distortion': np.mean(spectral_dist_values) if spectral_dist_values else None,
            'CHD': np.mean(chd_values) if chd_values else None,
            'FSIMc': np.mean(fsim_values) if fsim_values else None,  # Changed from FSIM to FSIMc
            'CW_SSIM': np.mean(cwssim_values) if cwssim_values else None
        },
        'detailed_metrics': {
            'sam_values': sam_values,
            'ergas_values': ergas_values,
            'spectral_dist_values': spectral_dist_values,
            'chd_values': chd_values,
            'fsim_values': fsim_values,
            'cwssim_values': cwssim_values
        }
    }
    
    # Save results to log file
    if save_log:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"metrics_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        for key in final_results['detailed_metrics']:
            if isinstance(final_results['detailed_metrics'][key], list):
                final_results['detailed_metrics'][key] = [float(x) for x in final_results['detailed_metrics'][key]]
        
        with open(log_filename, 'w') as f:
            json.dump(final_results, f, indent=4)
            
        # Also save a simple text summary
        summary_filename = f"metrics_summary_{timestamp}.txt"
        with open(summary_filename, 'w') as f:
            f.write("Metrics Evaluation Summary\n")
            f.write("=========================\n\n")
            f.write(f"Synthesized Folder: {synthesized_folder}\n")
            f.write(f"Ground Truth Folder: {ground_truth_folder}\n")
            f.write(f"Evaluation Time: {final_results['metadata']['timestamp']}\n")
            f.write(f"Number of Images: {final_results['metadata']['num_images_processed']}\n\n")
            f.write("Metric Results:\n")
            f.write("-------------\n")
            for metric_name, value in final_results['metrics'].items():
                if value is not None:
                    f.write(f"Mean {metric_name}: {value:.4f}\n")
                else:
                    f.write(f"No valid {metric_name} scores computed.\n")
    
    return final_results

def compute_chd(x, y, bins=256):
    """Compute Color Histogram Distance using OpenCV with Bhattacharyya Distance."""
    if torch.is_tensor(x):
        # Move tensors to CPU before converting to NumPy
        x = x.cpu()
        y = y.cpu()
        x = (x.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        y = (y.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    chd_values = []
    for channel in range(3):
        hist1 = cv2.calcHist([x], [channel], None, [bins], [0, 256])
        hist2 = cv2.calcHist([y], [channel], None, [bins], [0, 256])
        
        # Normalize histograms
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        chd_values.append(cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA))
    
    return np.mean(chd_values)

if __name__ == "__main__":
    synthesized_folder = "synthesized_folder_path"
    ground_truth_folder = "ground_truth_folder_folder_path"
    
    results = compute_metrics_for_folders(synthesized_folder, ground_truth_folder)
    
    print("\nMetric Results:")
    for metric_name, value in results['metrics'].items():
        if value is not None:
            print(f"Mean {metric_name}: {value:.4f}")
        else:
            print(f"No valid {metric_name} scores computed.")