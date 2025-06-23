import numpy as np
import os
import cv2
import shutil
import shutil
from PIL import Image
import json
import fiona
import rasterio
import rasterio.mask
import glob
import subprocess
from pathlib import Path
from scipy import fftpack
from skimage import io 
from skimage import exposure
import skimage.color as color
from scipy.ndimage import uniform_filter
from scipy.ndimage import variance
import cv2

def naming(source_dir, target_dir, word):
    """
    Copy files from source directory to target directory, renaming them to start from the specified word.
    """
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    for file in os.listdir(source_dir):
        shutil.copy(Path(source_dir) / file, Path(target_dir) / file[file.find(word):])

def count_zero(vector):
    return np.sum(vector == 0)

def extract(RGB_dir, RGB_target_dir, target_height, target_width, extraction_factor):
    """
    Extracts smaller image patches from larger images (RGB, SAR).
    For each dimension, if the leftover (remainder) after dividing by the target size
    is greater than or equal to extraction_factor * target size, an additional patch is extracted.

    Args:
        RGB_dir (str): Directory containing RGB images.
        RGB_target_dir (str): Directory to save extracted RGB patches.
        target_height (int): Height of the extracted patches.
        target_width (int): Width of the extracted patches.
        extraction_factor (float): If the leftover region is greater than or equal to this factor times the target size, extract an additional patch.
    """
    # Define directories for SAR images
    SAR_dir = RGB_dir.replace("PS-RGB", "SAR-Intensity")
    SAR_target_dir = RGB_target_dir.replace("PS-RGB", "SAR-Intensity")

    # Create target directories if they don't exist
    for dir in [RGB_target_dir, SAR_target_dir]:
        os.makedirs(dir, exist_ok=True)

    # Get file lists for each image type
    RGB_filelist = os.listdir(RGB_dir)
    SAR_filelist = os.listdir(SAR_dir)

    total_files = len(RGB_filelist)

    for idx_file in range(total_files):
        # Read RGB and SAR images
        RGB_arr = cv2.imread(f"{RGB_dir}/{RGB_filelist[idx_file]}")
        SAR_arr = cv2.imread(f"{SAR_dir}/{SAR_filelist[idx_file]}")

        image_shape = [int(RGB_arr.shape[0]), int(RGB_arr.shape[1])]
        target_shape = [target_height, target_width]
        quotient = [image_shape[0] // target_shape[0], image_shape[1] // target_shape[1]]
        remainder = [image_shape[0] % target_shape[0], image_shape[1] % target_shape[1]]

        # Determine the number of patches in each dimension
        num_patches = []
        for i in range(2):
            # Always at least one patch
            n = quotient[i]
            # If leftover is large enough, add one more patch
            if remainder[i] >= extraction_factor * target_shape[i]:
                n += 1
            num_patches.append(n)

        try:
            for h in range(num_patches[0]):
                for w in range(num_patches[1]):
                    # Calculate start and end coordinates for extraction
                    start_h = h * target_shape[0]
                    start_w = w * target_shape[1]
                    end_h = start_h + target_shape[0]
                    end_w = start_w + target_shape[1]

                    # If patch goes out of bounds, shift window to fit
                    if end_h > image_shape[0]:
                        start_h = image_shape[0] - target_shape[0]
                        end_h = image_shape[0]
                    if end_w > image_shape[1]:
                        start_w = image_shape[1] - target_shape[1]
                        end_w = image_shape[1]

                    # Extract sub-images
                    RGB_patch = RGB_arr[start_h:end_h, start_w:end_w, :]
                    SAR_patch = SAR_arr[start_h:end_h, start_w:end_w, :]

                    # Save extracted sub-images
                    for arr, dir, filelist in zip([RGB_patch, SAR_patch],
                                                  [RGB_target_dir, SAR_target_dir],
                                                  [RGB_filelist, SAR_filelist]):
                        filename = f"{filelist[idx_file].split('.png')[0]}_{start_h}_{start_w}.png"
                        cv2.imwrite(f"{dir}/{filename}", arr)

        except Exception as e:
            print(f"Error processing file {RGB_filelist[idx_file]}: {str(e)}")

    print("Extraction completed.")
            
            
def get_tile_name(filename, word="tile_"):
    idx = filename.find(word)
    return filename[idx:] if idx != -1 else filename

def rgb_to_png(RGB_dir, RGB_png_dir):
    os.makedirs(RGB_png_dir, exist_ok=True)
    for RGB_file_name in os.listdir(RGB_dir):
        if not RGB_file_name.endswith('.tif'):
            continue
        img = cv2.imread(os.path.join(RGB_dir, RGB_file_name), cv2.IMREAD_UNCHANGED)
        img = ((img - np.min(img)) / (np.max(img) - np.min(img))) * 255
        png_name = get_tile_name(RGB_file_name, "tile_").replace(".tif", ".png")
        cv2.imwrite(os.path.join(RGB_png_dir, png_name), img.astype(np.uint8))
    print("rgb_to_png conversion completed.")

def sar_to_HHVVVH_bgr2rgb(SAR_dir, SAR_png_dir):
    os.makedirs(SAR_png_dir, exist_ok=True)
    for SAR_file_name in os.listdir(SAR_dir):
        if not SAR_file_name.endswith('.tif'):
            continue
        tif_img = cv2.imread(os.path.join(SAR_dir, SAR_file_name), cv2.IMREAD_UNCHANGED)
        for channel in [0, 3, 2]:  # HH, VV, VH channels
            tif_img[:,:,channel] = ((tif_img[:,:,channel] - np.min(tif_img[:,:,channel])) / 
                                    (np.max(tif_img[:,:,channel]) - np.min(tif_img[:,:,channel]))) * 255
        img = np.concatenate((tif_img[:,:,0].reshape(900,900,1),
                              tif_img[:,:,3].reshape(900,900,1),
                              tif_img[:,:,2].reshape(900,900,1)), axis=2)
        img_rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        png_name = get_tile_name(SAR_file_name, "tile_").replace(".tif", ".png")
        cv2.imwrite(os.path.join(SAR_png_dir, png_name), img_rgb)
    print("sar_to_HHVVVH and BGR2RGB conversion completed.")

def rename(dir, target_dir, word):
    """
    Copy files from source directory to target directory, renaming them to start from a specified word.

    Args:
    dir (str): Source directory
    target_dir (str): Target directory
    word (str): Word to start the new filename from
    """
    os.makedirs(target_dir, exist_ok=True)
    
    for file in os.listdir(dir):
        shutil.copy(os.path.join(dir, file), os.path.join(target_dir, file[file.find(word):]))
    print(f"All files under {dir} are renamed.")

def cal_min_max_coord(json_dir):
    """
    Calculate min and max coordinates from all JSON files in the given directory.

    Args:
    json_dir (str): Directory containing JSON files with corner coordinates

    Returns:
    tuple: (top_left, top_right, bottom_left, bottom_right) coordinates
    """
    corners = {
        "top_left": [], "top_right": [],
        "bottom_left": [], "bottom_right": []
    }
    
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            with open(os.path.join(json_dir, filename)) as json_file:
                data = json.load(json_file)
            
            # Extract corner coordinates
            corners["top_left"].append(data["cornerCoordinates"]["upperLeft"])
            corners["top_right"].append(data["cornerCoordinates"]["upperRight"])
            corners["bottom_left"].append(data["cornerCoordinates"]["lowerLeft"])
            corners["bottom_right"].append(data["cornerCoordinates"]["lowerRight"])
    
    # Calculate min and max coordinates
    max_x = max(max(item[0] for item in corners["top_right"]), max(item[0] for item in corners["bottom_right"]))
    min_x = min(min(item[0] for item in corners["top_left"]), min(item[0] for item in corners["bottom_left"]))
    max_y = max(max(item[1] for item in corners["top_left"]), max(item[1] for item in corners["top_right"]))
    min_y = min(min(item[1] for item in corners["bottom_left"]), min(item[1] for item in corners["bottom_right"]))
    
    return (min_x, max_y), (max_x, max_y), (min_x, min_y), (max_x, min_y)

def get_coord_from_json(json_path):
    """
    Extract corner coordinates from a JSON file.

    Args:
    json_path (str): Path to the JSON file

    Returns:
    tuple: (top_left, top_right, bottom_left, bottom_right) coordinates
    """
    with open(json_path) as json_file:
        data = json.load(json_file)
    
    return (data["cornerCoordinates"]["upperLeft"],
            data["cornerCoordinates"]["upperRight"],
            data["cornerCoordinates"]["lowerLeft"],
            data["cornerCoordinates"]["lowerRight"])

def convert_coord_for_ref_png(ref_png, ref_coords, subject_coords):
    """
    Convert geographical coordinates to pixel coordinates relative to a reference image.

    Args:
    ref_png (str or numpy.ndarray): Reference image or path to reference image
    ref_coords (str or tuple): Reference coordinates or path to reference JSON file
    subject_coords (str or tuple): Subject coordinates or path to subject JSON file

    Returns:
    tuple: (top_left_pixel, top_right_pixel, bottom_left_pixel, bottom_right_pixel)
    """
    # Load reference image if a path is provided
    if isinstance(ref_png, str):
        ref_png = cv2.imread(ref_png, cv2.IMREAD_UNCHANGED)
    
    image_height, image_width = ref_png.shape[:2]
    
    # Get reference coordinates
    if isinstance(ref_coords, str):
        ref_coords = get_coord_from_json(ref_coords)
    ref_top_left, ref_top_right, ref_bottom_left, ref_bottom_right = ref_coords
    
    # Get subject coordinates
    if isinstance(subject_coords, str):
        subject_coords = get_coord_from_json(subject_coords)
    subject_top_left, subject_top_right, subject_bottom_left, subject_bottom_right = subject_coords
    
    # Convert geographical coordinates to pixel coordinates
    def geo_to_pixel(coord):
        x = int((coord[0] - ref_bottom_left[0]) * 2)
        y = image_height - int((coord[1] - ref_bottom_left[1]) * 2)
        return x, y
    
    return tuple(map(geo_to_pixel, [subject_top_left, subject_top_right, subject_bottom_left, subject_bottom_right]))

def extract_json(tif_dir, json_extracted_dir):
    os.makedirs(json_extracted_dir, exist_ok=True)
    tif_files = [f for f in os.listdir(tif_dir) if f.endswith('.tif')]
    for tif_file in tif_files:
        base_filename = get_tile_name(tif_file, "tile_").replace('.tif', '')
        output_json = os.path.join(json_extracted_dir, f"{base_filename}.json")
        command = ["gdalinfo", "-json", os.path.join(tif_dir, tif_file)]
        with open(output_json, "w") as output_file:
            subprocess.run(command, stdout=output_file, text=True)
    print(f"Extraction completed. Extracted JSON files saved in: {json_extracted_dir}")

def split(RGB_json_dir, div_line_coords):
    save_path = RGB_json_dir.replace("extract_json", "split")
    train_save_path = os.path.join(save_path, "train")
    val_save_path = os.path.join(save_path, "val")
    test_save_path = os.path.join(save_path, "test")
    overlapped_save_path = os.path.join(save_path, "overlapped_with_line")
    for path in [save_path, train_save_path, val_save_path, test_save_path, overlapped_save_path]:
        os.makedirs(path, exist_ok=True)
        os.makedirs(path.replace("PS-RGB", "SAR-Intensity"), exist_ok=True)
    div_line_1_x = div_line_coords[0][0]
    div_line_2_x = div_line_coords[2][0]
    with open(os.path.join(save_path, "train&test_split_by_geo_coords.txt"), "w") as f:
        f.write(f"Division Line 1 Geo X-Coordinate:\t{div_line_1_x}\n")
        f.write(f"Division Line 2 Geo X-Coordinate:\t{div_line_2_x}\n\n")
        for filename in os.listdir(RGB_json_dir):
            if not filename.endswith(".json"):
                continue
            json_path = os.path.join(RGB_json_dir, filename)
            data_coords = get_coord_from_json(json_path)
            tile_top_left_x = data_coords[0][0]
            tile_top_right_x = data_coords[1][0]
            if tile_top_right_x <= div_line_1_x:
                set_type = "train"
                target_path = train_save_path
            elif div_line_1_x < tile_top_left_x <= div_line_2_x:
                set_type = "val"
                target_path = val_save_path
            elif div_line_2_x < tile_top_left_x:
                set_type = "test"
                target_path = test_save_path
            else:
                set_type = "overlapped"
                target_path = overlapped_save_path
            f.write(f"{filename}\tleft_x:{tile_top_left_x}\tright_x:{tile_top_right_x}\t-> {set_type}\n")
            base_filename = filename.replace('.json', '.png')
            for modality in ["PS-RGB", "SAR-Intensity"]:
                src = f"{RGB_json_dir.replace('extract_json', 'png').replace('PS-RGB', modality)}/{base_filename}"
                dst = f"{target_path.replace('PS-RGB', modality)}/{base_filename}"
                shutil.copy2(src, dst)
    print("Train & test split is done.")

def cut_zero_png(input_dir, output_dir, black_rate=0.95):
    """
    Remove rows and columns with a high percentage of black pixels from PNG images.
    Args:
        input_dir (str): Directory containing input PNG images.
        output_dir (str): Directory to save processed PNG images.
        black_rate (float): Threshold ratio for black pixels to remove row/column.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_list = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    for file_name in file_list:
        img = cv2.imread(os.path.join(input_dir, file_name))
        if img is None:
            print(f"Failed to read {file_name}")
            continue

        # Remove rows with too many black pixels
        row_black = np.sum(np.sum(img, axis=2) == 0, axis=1) > int(black_rate * img.shape[1])
        img = img[~row_black]

        # Remove columns with too many black pixels
        col_black = np.sum(np.sum(img, axis=2) == 0, axis=0) > int(black_rate * img.shape[0])
        img = img[:, ~col_black]

        cv2.imwrite(os.path.join(output_dir, file_name), img)

def processing(
    orig_dir, 
    div_line_coords, 
    target_height=512, 
    target_width=512, 
    extraction_factor=0.25,
    use_cut_zero=True,
    black_rate=0.95
):
    """
    Main processing pipeline for dataset preparation.
    Args:
        orig_dir (str): Root directory containing original data.
        div_line_coords (tuple): Coordinates for train/val/test split.
        target_height (int): Height of patches to extract.
        target_width (int): Width of patches to extract.
        extraction_factor (float): Extraction factor for patching.
        use_cut_zero (bool): Whether to apply cut_zero_png after PNG conversion.
        black_rate (float): Threshold for cut_zero_png.
    """
    RGB_dir = os.path.join(orig_dir, "PS-RGB")
    SAR_dir = os.path.join(orig_dir, "SAR-Intensity")
    dirs = [RGB_dir, SAR_dir]
    processed_dirs = [dir + "_processed" for dir in dirs]
    RGB_dir_dict = {}
    SAR_dir_dict = {}
    dicts = [RGB_dir_dict, SAR_dir_dict]
    processing_keywords = ["extract_json", "png", "split"]
    for keyword in processing_keywords:
        for idx in range(len(dicts)):
            dicts[idx][keyword] = os.path.join(processed_dirs[idx], keyword)
            os.makedirs(dicts[idx][keyword], exist_ok=True)

    # 1. Extract JSON metadata
    extract_json(RGB_dir, dicts[0]["extract_json"])
    extract_json(SAR_dir, dicts[1]["extract_json"])

    # 2. Convert to PNG
    rgb_to_png(RGB_dir, dicts[0]["png"])
    sar_to_HHVVVH_bgr2rgb(SAR_dir, dicts[1]["png"])

    # 2-1. Apply cut_zero to PNGs (optional)
    if use_cut_zero:
        cutzero_dirs = [dicts[0]["png"] + "_cutzero", dicts[1]["png"] + "_cutzero"]
        cut_zero_png(dicts[0]["png"], cutzero_dirs[0], black_rate)
        cut_zero_png(dicts[1]["png"], cutzero_dirs[1], black_rate)
    else:
        cutzero_dirs = [dicts[0]["png"], dicts[1]["png"]]

    # 3. Extract patches from PNG images (before split)
    # Output directory for RGB patches
    RGB_patch_dir = cutzero_dirs[0] + f"_patches_{target_height}x{target_width}"
    # Output directory for SAR patches
    SAR_patch_dir = cutzero_dirs[1] + f"_patches_{target_height}x{target_width}"

    # 4. Split using JSON from original images (not patches)
    RGB_json_dir = dicts[0]["extract_json"]
    split(RGB_json_dir, div_line_coords)

    # 5. Apply cut_zero to each split directory
    split_names = ["train", "val", "test"]
    for modality in ["PS-RGB", "SAR-Intensity"]:
        for split_name in split_names:
            split_dir = os.path.join(f"{modality}_processed", "split", split_name)
            cutzero_dir = split_dir + "_cutzero"
            cut_zero_png(split_dir, cutzero_dir, black_rate)

    # 6. After cut_zero, extract patches from cutzero split images
    for split_name in split_names:
        for modality in ["PS-RGB", "SAR-Intensity"]:
            input_dir = os.path.join(f"{modality}_processed", "split", split_name + "_cutzero")
            output_dir = input_dir + f"_patches_{target_height}x{target_width}"
            extract(
                RGB_dir=input_dir,
                RGB_target_dir=output_dir,
                target_height=target_height,
                target_width=target_width,
                extraction_factor=extraction_factor
            )

    print("All processing done.")

def clahe_sar_png_dir(sar_png_dir):
    
    clahe = cv2.createCLAHE(clipLimit=2)
    sar_png_files = glob.glob(os.path.join(sar_png_dir, "*.png"))
    
    for sar_png_file in sar_png_files:
        img = cv2.imread(sar_png_file)
        clahe_img = np.zeros_like(img)
        
        h,w,channels = img.shape
        for channel in range(channels):
            clahe_img[:,:,channel] = clahe.apply(img[:,:,channel])
        
        cv2.imwrite(sar_png_file, clahe_img)
    

if __name__ == "__main__":    
    processing("/path/to/AOI_11_Rotterdam", ((595788.885371555, 5753195.131), (595788.885371555, 5753195.131), (596141.3430362847, 5745470.425), (596141.3430362847, 5745470.425)))