#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import csv
import glob
from datetime import datetime
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from sys import getsizeof
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

iter_num = 30100 # Number of iterations to run (based on SH 875 experiments, best quality at ~16k; verify on other datasets)

# [4000, 3000, 2500, 2000, 1500, 1250, 1000, 750, 500]
# [3000 2000 1500 1000 750 500]
sh_num = 1000 # 

common_iterations = [i for i in range(200, iter_num, 200)]

test_iterations = common_iterations + [iter_num]
save_iterations = common_iterations + [iter_num]

def measure_model_sizes(output_dir):
    model_sizes = {}
    model_dir_pattern = os.path.join(output_dir, "Model*", "point_cloud", "iteration_*")
    model_dirs = glob.glob(model_dir_pattern)

    for model_dir in model_dirs:
        iteration = os.path.basename(model_dir).split('_')[-1]
        ply_files = glob.glob(os.path.join(model_dir, "*.ply"))

        total_size = 0 
        for ply_file in ply_files:
            total_size += os.path.getsize(ply_file)

        model_sizes[iteration] = total_size / 1024 / 1024  # Convert to MB

    return model_sizes

def update_csv_with_model_sizes(csv_path, model_sizes):
    with open(csv_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        lines = list(reader)

    header = lines[0]
    if len(header) < 8:  
        header.extend([''] * (8 - len(header)))  
    header[7] = 'Model Size (MB)'

    for i, line in enumerate(lines[1:]):  # Skip header
        iteration = line[0]  
        if iteration in model_sizes:
            if len(line) < 8:  
                line.extend([''] * (8 - len(line)))
            line[7] = str(model_sizes[iteration])  

    # Write updated data back to CSV
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(lines)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    accumulated_time = 0.0  # Initialize accumulated time

    # Initialize CSV file and writer
    csv_path = r'[\gaussian-splatting\CSVData\training_data.csv]'
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Iteration', 'SSIM', 'L1', 'PSNR', 'Loss', 'FPS', 'Iteration Time'])

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, min(opt.iterations, iter_num+1)), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, min(opt.iterations, iter_num+1)):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % sh_num == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_val = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        torch.cuda.synchronize()
        elapsed_time = iter_start.elapsed_time(iter_end) / 1000  # Elapsed time in seconds
        accumulated_time += elapsed_time  # Accumulate the elapsed time
        formatted_accumulated_time = "{:.3f}".format(accumulated_time)

        fps = 1 / elapsed_time

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 1 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(1)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if iteration in testing_iterations:
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, formatted_accumulated_time, testing_iterations, scene, render, (pipe, background), csv_writer, ssim_val.item(), fps, elapsed_time)
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        # Get the current system time and format it as you like
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Modify the model_path format here
        args.model_path = os.path.join("./output/", f"Model{current_time}")
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, formatted_accumulated_time, testing_iterations, scene, renderFunc, renderArgs, csv_writer, ssim, fps, elapsed_time):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed_time, iteration)  # Use elapsed_time here


    psnr_test = 0.0  # Initialize PSNR variable

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
       
        # Ensure PSNR is a Python float and not a tensor
        psnr_value = psnr_test.item() if torch.is_tensor(psnr_test) else psnr_test

        # Update CSV file
        csv_writer.writerow([iteration, ssim, Ll1.item(), psnr_value, loss.item(), fps, formatted_accumulated_time])

        if tb_writer:
            tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
            tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=test_iterations)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=test_iterations)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # Run training
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")

    # Update CSV with model sizes
    model_sizes = measure_model_sizes(r'[\gaussian-splatting\output]')
    csv_path = r'[\gaussian-splatting\CSVData\training_data.csv]'
    update_csv_with_model_sizes(csv_path, model_sizes)

    # Compute and display final SSIM value
    final_ssim = None
    csv_path = r'[\gaussian-splatting\CSVData\training_data.csv]'

    # Read the last SSIM value from the CSV
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            final_ssim = row[1]  # SSIM value is in the second column

    if final_ssim:
        print(f"\nModel SSIM: {final_ssim}")


