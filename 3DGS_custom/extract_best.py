# MEIC Lab Frame Extraction Optimizer
# Hansol Lim 2024-03-20
# hansol.lim@stonybrook.edu

import cv2
import numpy as np
import os
import shutil
import time
import sys
import random
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from contextlib import redirect_stdout


def read_config():
    config = { 
        'overlap_threshold': 85,   # Overlap threshold percentage
        'warning_attempts': 3,     # Number of retry warnings for frame extraction
        'accelerated_optimization': 0,  # Enable accelerated sampling optimization (may reduce accuracy)
        'sampling_percentage': 25, # Percentage for random sampling in optimization
    }

    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    else:
        application_path = os.path.dirname(__file__)

    config_path = os.path.join(application_path, 'config.txt')

    try:
        with open(config_path, 'r') as file:
            for line in file:
                key, value = line.strip().split('=')
                if key in config:
                    config[key] = float(value) if '.' in value else int(value)
    except FileNotFoundError:
        print(f"config.txt not found. Running with default settings.")

    return config

def save_config(config):
    config_path = "config.txt"
    try:
        with open(config_path, 'w', encoding='utf-8') as file:
            for key, value in config.items():
                file.write(f"{key}={value}\n")
        messagebox.showinfo("Saving complete", "Settings have been saved successfully.")
    except Exception as e:
        messagebox.showerror("Saving Error", f"Failed to save settings: {e}")


def open_config_editor():
    for key in config:
        configmessages = {
            'overlap_threshold': '(default:85)',
            'warning_attempts': '(default:3)',
            'accelerated_optimization': '(default:0 off, 1 on)',
            'sampling_percentage': '(default: 25%)',
        }
        
        new_value = simpledialog.askstring("Change Settings", f"{configmessages[key]} (Current Value: {config[key]})", parent=root)
        if new_value is not None:
            try:
                config[key] = int(new_value) if new_value.isdigit() else float(new_value)
            except ValueError:
                messagebox.showerror("Error", "Invalide Value. Try again.")
                return
    
    save_config(config)
    root.deiconify()


def upload_video_and_process():
    root.withdraw()
    video_path = filedialog.askopenfilename(title="Select Video File", filetypes=(("Video files", "*.mp4 *.avi *.MOV"), ("All files", "*.*")), parent=root)
    if video_path:
        print(f"Filename: {os.path.basename(video_path)}")
        output_folder = os.path.splitext(video_path)[0]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        print(f"Saving frames to: {output_folder}")
        main(video_path, output_folder, config)
        announce_moved_images(output_folder)
        messagebox.showinfo("Done", "Processing complete.", parent=root)
    else:
        messagebox.showwarning("Warning", "Please select a video file (.mp4, .avi, .MOV).", parent=root)
    
    root.deiconify()
    end_timer()

def downsample_video(video_path, output_path, target_height=720):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video file.")
        return False
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    target_width = int((target_height / height) * width)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

    print("Downsampling video...")
    with tqdm(total=total_frames, desc="Progress") as pbar:
        while True:
            ret, frame = cap.read()
            if ret:
                resized_frame = cv2.resize(frame, (target_width, target_height))
                out.write(resized_frame)
                pbar.update(1)
            else:
                break

    cap.release()
    out.release()
    return True


def heuristic_model_frames(video_length):  # Frame extractor based on heuristic model
    if video_length <= 250:
        num_frames = 920 + 2 * video_length
    elif video_length <= 1200:
        a, b, c, d, e = 2.68392e-08, -8.58658e-05, 0.098846, -46.00985, 7993.18884  
        num_frames = (a * video_length**4) + (b * video_length**3) + (c * video_length**2) + (d * video_length) + e
    else:
        num_frames = video_length * 2
    return max(int(round(num_frames)), 1)


def save_frames_dur(cap_local, total_frames_to_save):
    video_length = cap_local.get(cv2.CAP_PROP_FRAME_COUNT) / cap_local.get(cv2.CAP_PROP_FPS)
    return np.linspace(0, video_length, total_frames_to_save, endpoint=False)


def save_frame(frame, folder_name, img_count):
    cv2.imwrite(os.path.join(folder_name, f"img{img_count:04}.jpg"), frame)


def overlap_warp_percentage(warped_image, target_image):
    gray_warped = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    _, binary_warped = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)
    _, binary_target = cv2.threshold(gray_target, 1, 255, cv2.THRESH_BINARY)
    overlap = cv2.bitwise_and(binary_warped, binary_target)
    non_zero_overlap = np.count_nonzero(overlap)
    non_zero_target = np.count_nonzero(binary_target)
    overlap_percentage = (non_zero_overlap / non_zero_target) * 100
    return overlap_percentage


def homography_overlap_estimator(image_path1, image_path2, output_folder):  # Homography Matrix Computation
    if not os.path.exists(image_path1) or not os.path.exists(image_path2):
        return None

    image1, image2 = cv2.imread(image_path1), cv2.imread(image_path2)
    gray1, gray2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 4:  
        move_to_featureless_images(output_folder, os.path.basename(image_path1))
        move_to_featureless_images(output_folder, os.path.basename(image_path2))
        return None

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros_like(points1)
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC) 
    if h is None: 
        move_to_featureless_images(output_folder, os.path.basename(image_path1))
        move_to_featureless_images(output_folder, os.path.basename(image_path2))
        return None

    height, width, channels = image2.shape
    warped_image1 = cv2.warpPerspective(image1, h, (width, height))
    return overlap_warp_percentage(warped_image1, image2)


def accelerated_overlap_threshold_checker(output_folder, overlap_threshold, sampling_percentage):
    featureless_problem_frames = []
    image_paths = sorted([os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.jpg')])
    num_images = len(image_paths)
    sample_size = max(int(num_images * sampling_percentage / 100), 1)
    sampled_indices = random.sample(range(num_images - 1), sample_size)

    valid_overlaps = []
    moved_images_count = 0

    for i in tqdm(sampled_indices, desc=(f"Calculating overlap (random sampling {sampling_percentage}%)"):
        overlap_percentage = homography_overlap_estimator(image_paths[i], image_paths[i + 1], output_folder)
        if overlap_percentage is not None:
            valid_overlaps.append(overlap_percentage)
        else:
            moved_images_count += 2

    if valid_overlaps:
        average_overlap = sum(valid_overlaps) / len(valid_overlaps)
        overlaps_met_criteria = average_overlap >= overlap_threshold
    else:
        print("No valid overlap data available.")
        average_overlap = 0
        overlaps_met_criteria = False

    return average_overlap, overlaps_met_criteria, moved_images_count, featureless_problem_frames


def overlap_threshold_checker(output_folder, overlap_threshold):  
    featureless_problem_frames = []
    image_paths = sorted([os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.jpg')])
    valid_overlaps = []
    moved_images_count = 0

    pbar = tqdm(total=len(image_paths) - 1, desc="Calculating overlap")
    for i in range(len(image_paths) - 1):
        overlap_percentage = homography_overlap_estimator(image_paths[i], image_paths[i + 1], output_folder)
        if overlap_percentage is not None:
            valid_overlaps.append(overlap_percentage)
        else:
            moved_images_count += 2
        pbar.update(1)
    pbar.close()

    if valid_overlaps:
        average_overlap = sum(valid_overlaps) / len(valid_overlaps)
        overlaps_met_criteria = average_overlap >= overlap_threshold
    else:
        print("No valid overlap data available.")
        average_overlap = 0
        overlaps_met_criteria = False

    return average_overlap, overlaps_met_criteria, moved_images_count, featureless_problem_frames


def delete_existing_images(output_folder, exclude_files=[]):
    for item in os.listdir(output_folder):
        item_path = os.path.join(output_folder, item)
        if item_path not in exclude_files:
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)


def frame_extractor(video_path, output_folder, frame_ex_multiplier):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video file.")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_length = total_frames / fps

    base_frame_count = heuristic_model_frames(video_length)
    target_frame_count = int(base_frame_count * frame_ex_multiplier)

    print(f"Target frame count: {target_frame_count}, multiplier: {frame_ex_multiplier:.4f}")

    frames_to_extract = max(int(total_frames / target_frame_count), 1)
    extracted_count = 0

    with tqdm(total=min(target_frame_count, total_frames), desc="Extracting frames", unit="frame") as pbar:
        for i in range(0, total_frames, frames_to_extract):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Save frames with sequential numbering
                frame_filename = f"{extracted_count + 1:04}.jpg"
                cv2.imwrite(os.path.join(output_folder, frame_filename), frame)
                extracted_count += 1
                pbar.update(1)
                if extracted_count >= target_frame_count:
                    break

    cap.release()
    print(f"{extracted_count} frames have been extracted.")
    return True


def laplacian_blur_filter(output_folder):
    blur_problem_frames = []
    image_paths = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.jpg')]

    for path in tqdm(image_paths, desc="Calculating blurriness"):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()

        if laplacian_var < 50:
            blur_problem_frames.append(os.path.basename(path))

    print(f"Warning: {len(blur_problem_frames)} blurry images detected. All images exceed threshold; none were filtered.")

    if not blur_problem_frames:
        print("All images exceed threshold; none were filtered.")
        
    return blur_problem_frames



def move_to_blurry_images(folder_name, img_name):
    blurry_images_folder = os.path.join(folder_name, "blurry_images")
    if not os.path.exists(blurry_images_folder):
        os.makedirs(blurry_images_folder)
    os.rename(os.path.join(folder_name, img_name), os.path.join(blurry_images_folder, img_name))

def move_to_featureless_images(folder_name, img_name):
    featureless_images = os.path.join(folder_name, "featureless_images")
    if not os.path.exists(featureless_images):
        os.makedirs(featureless_images)
    os.rename(os.path.join(folder_name, img_name), os.path.join(featureless_images, img_name))

def analyze_downsampled_video(output_folder, overlap_threshold, sampling_percentage):
    overlap_problem_frames = accelerated_overlap_threshold_checker(output_folder, overlap_threshold, sampling_percentage)
    blur_problem_frames = laplacian_blur_filter(output_folder)
    return list(set(overlap_problem_frames + blur_problem_frames))

def update_multiplier(multiplier, iteration):
    if iteration < 4:
        return multiplier + 0.1
    else:
        return multiplier * (1.1 + (iteration - 3) * 0.02)


def reorder_images(output_folder):  
    def extract_number(filename):
        return int(''.join(filter(str.isdigit, filename)) or -1)

    image_files = sorted(
        [f for f in os.listdir(output_folder) if f.endswith(('.png', '.jpg', '.jpeg'))],
        key=extract_number
    )
    for i, file in enumerate(tqdm(image_files, desc="Resampling"), start=1):
        os.rename(os.path.join(output_folder, file), os.path.join(output_folder, f"{i:04}.jpg"))
    print("Resampling complete")



def announce_moved_images(output_folder):
    featureless_images_folder = os.path.join(output_folder, "featureless_images")
    blurry_images_folder = os.path.join(output_folder, "blurry_images")
    
    featureless_images_count = 0
    blurry_images_count = 0
    
    if os.path.exists(featureless_images_folder):
        featureless_images_count = len([name for name in os.listdir(featureless_images_folder) if os.path.isfile(os.path.join(featureless_images_folder, name))])
    
    if os.path.exists(blurry_images_folder):
        blurry_images_count = len([name for name in os.listdir(blurry_images_folder) if os.path.isfile(os.path.join(blurry_images_folder, name))])
    
    total_moved_images = featureless_images_count + blurry_images_count
    
    if total_moved_images > 0:
        print(f"{count_feature} images moved to 'featureless_images' due to lack of features.")
        print(f"{count_blurry} images moved to 'blurry_images' due to blurriness.")
        print(f"A total of {total_moved} images were processed.")
    else:
        print("No images were moved.")

        
def resample_frames(output_folder, target_frame_count=300):
    image_files = sorted(
        [f for f in os.listdir(output_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    )
    total_frames = len(image_files)

    if total_frames <= target_frame_count:
        print(f"No need for resampling. Total frames: {total_frames}")
        return

    interval = max(total_frames // target_frame_count, 1)
    selected_files = [image_files[i] for i in range(0, total_frames, interval)][:target_frame_count]

    print(f"{total_frames} frames {target_frame_count} resampling...")
    for file in image_files:
        if file not in selected_files:
            os.remove(os.path.join(output_folder, file))
    print(f"Resampling Done. Final frames: {len(selected_files)}")



def end_timer():
    root.destroy()
    n = 5
    print(f"In {n} second, MEIC Lab Frame Extraction Optimizer will shut down.", end='', flush=True)
    while n > 0:
        time.sleep(1)
        n -= 1
        print(f'In \r{n} second, MEIC Lab Frame Extraction Optimizer will shut down.', end='', flush=True)


def upload_video():
    video_path = filedialog.askopenfilename(title="Select Video File", filetypes=(("Video files", "*.mp4 *.avi *.MOV"), ("All files", "*.*")))
    if video_path:
        print(f"File name: {os.path.basename(video_path)}")
        output_folder = os.path.splitext(video_path)[0]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        print(f"Saving to: {output_folder}")
        main(video_path, output_folder, config)
        announce_moved_images(output_folder)
    else:
        print("Please select a video file (.mp4, .avi, .MOV).")
    

def open_config():
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    else:
        application_path = os.path.dirname(__file__)

    config_path = os.path.join(application_path, 'config.txt')
    try:
        if sys.platform.startswith('darwin'):
            subprocess.run(['open', config_path], check=True)
        elif os.name == 'nt':  
            os.startfile(config_path)
        elif os.name == 'posix':  
            subprocess.run(['xdg-open', config_path], check=True)
    except Exception as e:
        print(f"Error opening config.txt: {e}")



def main(video_path, output_folder, config):
    print("Starting analysis...")
    overlap_threshold = config.get('overlap_threshold', 80)
    sampling_percentage = config.get('sampling_percentage', 0.2)
    warning_attempts = config.get('warning_attempts', 3)
    attempt = 1
    frame_ex_multiplier = 1.0
    downsampled_video_path = os.path.join(output_folder, "downsampled_video.mp4")
    blur_problem_frames = []
    featureless_problem_frames = []

    while True:
        print(f"{attempt}th trial: Initiating frame extraction...")

        if attempt == 1:
            if not os.path.exists(downsampled_video_path):
                print("Starting Downsampling...")
                if not downsample_video(video_path, downsampled_video_path, target_height=720):
                    print("Downsampling Failed.")
                    return
            else:
                print("Using previous downsampled video.")

        if not frame_extractor(downsampled_video_path, output_folder, frame_ex_multiplier):
            print("Frame extraction for previous downsampled video failed.")
            return

        average_valid_overlap, overlaps_met_criteria, _, featureless_temp = (
            accelerated_overlap_threshold_checker(output_folder, overlap_threshold, sampling_percentage)
            if config.get('accelerated_optimization', False)
            else overlap_threshold_checker(output_folder, overlap_threshold)
        )
        featureless_problem_frames.extend(featureless_temp)
        print(f"Average Overlap: {average_valid_overlap:.2f}%")

        if overlaps_met_criteria:
            print("Average overlap satifactory. Optimization Complete.")
            blur_problem_frames = laplacian_blur_filter(output_folder)
            print(f"Warning: {len(blur_problem_frames)} blurry images detected.")
            break

        if attempt >= warning_attempts:
            print(f"After optimization, average overlap is lower than {overlap_threshold}%. Processing more data.")
            break

        frame_ex_multiplier = update_multiplier(frame_ex_multiplier, attempt)
        print(f"Adjusting frame extraction: {frame_ex_multiplier:.4f}")

        attempt += 1
        delete_existing_images(output_folder, exclude_files=[downsampled_video_path])

    print("Extracting frames from original video...")
    frame_extractor(video_path, output_folder, frame_ex_multiplier)

    print("Sorting problematic frames...")
    for frame_name in blur_problem_frames:
        move_to_blurry_images(output_folder, frame_name)
    for frame_name in featureless_problem_frames:
        move_to_featureless_images(output_folder, frame_name)

    print("Resampling final 300 frames...")
    resample_frames(output_folder, target_frame_count=30)

    print("Resorting image file name after resampling...")
    reorder_images(output_folder)

    print(f"Analysis Complete. Blurry frames: {len(blur_problem_frames)}, Featureless images: {len(featureless_problem_frames)} are moved to each folder.")


if __name__ == "__main__":
    config = read_config()
    video_folder = r"[\gaussian-splatting\Uploads]"  
    output_base_folder = r"[\gaussian-splatting\Uploads]" 

    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.MOV'))]
    if not video_files:
        print(f"No file in '{video_folder}' to process.")
    else:
        for video_file in video_files:
            video_path = os.path.join(video_folder, video_file)
            print(f"Working on: {video_file}")

            output_folder = os.path.join(output_base_folder, "input")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            main(video_path, output_folder, config)

        print("All video process is complete.") 
