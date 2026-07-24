import time
import logging
from pathlib import Path
import datetime
import numpy as np

from options.config import load_config
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from util import util

# =========== Random seed setup ===========
import random
import numpy as np
import torch


# Configure logging - create separate logs for each checkpoint
def setup_checkpoint_logger(log_dir, checkpoint_name, run_id=None):
    """
    Set up logger for training, creating separate logs for each checkpoint
    run_id: unique identifier for each run to prevent overwriting
    """
    # Generate unique run ID using timestamp if not provided
    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create log path: log_dir/checkpoint_name/run_id.log
    log_path = Path(log_dir) / checkpoint_name / f"run_{run_id}.log"

    # Ensure all parent directories exist
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure log format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Create logger with unique name per checkpoint to avoid conflicts
    logger = logging.getLogger(f"training_{checkpoint_name}")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent log mixing
    if logger.handlers:
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(str(log_path))
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Log file created: {log_path}")
    return logger


def validate(model, val_dataloader, logger, epoch):
    """
    Validate the model on the validation set and compute PSNR and SSIM metrics
    """
    model.eval()

    # Initialize metrics dictionary
    metrics = {
        'fake_t1': {'psnr': [], 'ssim': []},
        'fake_t2': {'psnr': [], 'ssim': []},
        'fake_t1ce': {'psnr': [], 'ssim': []},
        'fake_flair': {'psnr': [], 'ssim': []}
    }

    logger.info(f"Starting validation for epoch {epoch}")

    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()

            # Calculate metrics for generated images
            for label, image in visuals.items():
                if 'fake_' in label:
                    real_key = label[5:]
                    if real_key in visuals:
                        real_img = visuals[real_key]
                        fake_img = visuals[label]

                        # Convert to numpy arrays
                        real_numpy = real_img.cpu().numpy()[0, 0]
                        fake_numpy = fake_img.cpu().numpy()[0, 0]

                        # Scale from [-1, 1] to [0, 255]
                        real_numpy = (real_numpy + 1) / 2 * 255
                        fake_numpy = (fake_numpy + 1) / 2 * 255

                        # Compute PSNR and SSIM
                        psnr_value = psnr(real_numpy, fake_numpy, data_range=255)
                        ssim_value = ssim(real_numpy, fake_numpy, data_range=255, multichannel=False, win_size=3)
                        ssim_value = ssim_value * 100

                        # Store metrics
                        if label in metrics:
                            metrics[label]['psnr'].append(psnr_value)
                            metrics[label]['ssim'].append(ssim_value)

    # Compute averages
    avg_metrics = {}
    for key in metrics.keys():
        if metrics[key]['psnr']:
            psnr_mean = np.mean(metrics[key]['psnr'])
            psnr_std = np.std(metrics[key]['psnr'])
            ssim_mean = np.mean(metrics[key]['ssim'])
            ssim_std = np.std(metrics[key]['ssim'])

            avg_metrics[key] = {
                'psnr_mean': psnr_mean,
                'psnr_std': psnr_std,
                'ssim_mean': ssim_mean,
                'ssim_std': ssim_std
            }

            logger.info(f'{key} - Mean PSNR: {psnr_mean:.2f}, Std PSNR: {psnr_std:.2f}')
            logger.info(f'{key} - Mean SSIM: {ssim_mean:.4f}, Std SSIM: {ssim_std:.4f}')

    model.train()

    # Return average metrics for model selection
    return avg_metrics


if __name__ == '__main__':
    # Load configuration
    opt = load_config()

    # Get checkpoint name (from config or use default)
    if hasattr(opt, 'checkpoints_dir'):
        checkpoint_name = Path(opt.checkpoints_dir).name  # Extract directory name as identifier
    else:
        checkpoint_name = "checkpoints_default"  # Default if not configured

    # Determine log root directory
    log_root = opt.log_dir if hasattr(opt, 'log_dir') else 'experiment_logs'

    # Initialize logger with unique run ID
    logger = setup_checkpoint_logger(log_root, checkpoint_name)

    # Load training dataset
    dataloader = create_dataset(opt)
    dataset_size = len(dataloader)
    logger.info(f"Number of training images: {dataset_size}")
    logger.info(f"Starting training for {opt.name} with checkpoint: {checkpoint_name}")

    # Load validation dataset
    val_opt = load_config()
    val_opt.__dict__.update(opt.__dict__)
    val_opt.isTrain = False
    val_opt.batch_size = 1
    val_opt.serial_batches = True
    val_opt.no_flip = True
    val_opt.display_id = -1
    val_dataloader = create_dataset(val_opt)
    val_dataset_size = len(val_dataloader)
    logger.info(f"Number of validation images: {val_dataset_size}")

    # Model initialization
    opt.n_input_modal = dataloader.dataset.n_modal - 1
    opt.modal_names = dataloader.dataset.get_modal_names()
    model = create_model(opt)
    model.setup(opt)
    logger.info(f"Model initialized: {opt.model}")

    # Handle SR model if present
    if hasattr(opt, 'sr_model') and isinstance(opt.sr_model, str):
        sr_opt = load_config(opt.sr_model)
        sr_opt.input_nc = opt.input_nc
        sr_opt.output_nc = opt.output_nc
        sr_opt.modal_names = opt.modal_names
        sr_model = create_model(sr_opt)
        sr_model.setup(sr_opt)
        model.add_srmodel(sr_model)
        logger.info(f"SR model added: {opt.sr_model}")

    total_iters = 0
    visualizer = Visualizer(opt, dataset_size)

    # Initialize best metrics for model selection (with std)
    best_psnr = 0.0
    best_psnr_ssim = 0.0
    best_psnr_std = 0.0
    best_ssim_std_for_psnr = 0.0
    best_psnr_epoch = 0

    best_ssim = 0.0
    best_ssim_psnr = 0.0
    best_ssim_std = 0.0
    best_psnr_std_for_ssim = 0.0
    best_ssim_epoch = 0

    best_combined = 0.0
    best_combined_psnr = 0.0
    best_combined_ssim = 0.0
    best_combined_psnr_std = 0.0
    best_combined_ssim_std = 0.0
    best_combined_epoch = 0

    # Training main loop
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):

        model.epoch = epoch

        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        logger.info(f"Starting epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay}")

        for i, data in enumerate(dataloader):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            # Display images if needed
            if total_iters % opt.display_freq == 0:
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), total_iters)

            # Print losses
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
                logger.info(
                    f"Epoch: {epoch}, Iteration: {epoch_iter}, Losses: {{{loss_str}}}, "
                    f"Computation time: {t_comp:.4f}, Data loading time: {t_data:.4f}"
                )
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

            # Save latest model
            if total_iters % opt.save_latest_freq == 0:
                logger.info(f"Saving latest model (epoch {epoch}, total iterations {total_iters})")
                save_suffix = f'iter_{total_iters}' if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            if total_iters % opt.save_latest_freq == 0:
                model.sr_weight *= 0.05

            iter_data_time = time.time()

        # Save latest model
        if epoch % opt.save_epoch_freq == 0:
            logger.info(f"Saving model at end of epoch {epoch}, total iterations {total_iters}")
            model.save_networks('latest')
            model.save_networks(epoch)

        # Validate after each epoch
        avg_metrics = validate(model, val_dataloader, logger, epoch)

        # Check if this is the best model based on t1ce metrics (since it's the target modality)
        if 'fake_t1ce' in avg_metrics:
            current_psnr = avg_metrics['fake_t1ce']['psnr_mean']
            current_ssim = avg_metrics['fake_t1ce']['ssim_mean']
            current_psnr_std = avg_metrics['fake_t1ce']['psnr_std']
            current_ssim_std = avg_metrics['fake_t1ce']['ssim_std']

            # Calculate combined score
            combined_score = 0.7 * (current_psnr / 50) + 0.3 * (current_ssim / 100)

            # Save best model based on PSNR
            if current_psnr > best_psnr:
                best_psnr = current_psnr
                best_psnr_ssim = current_ssim
                best_psnr_std = current_psnr_std
                best_ssim_std_for_psnr = current_ssim_std
                best_psnr_epoch = epoch
                logger.info(
                    f"New best PSNR model found! "
                    f"PSNR: {best_psnr:.2f}±{best_psnr_std:.2f}, "
                    f"SSIM: {best_psnr_ssim:.4f}±{best_ssim_std_for_psnr:.4f} "
                    f"at epoch {best_psnr_epoch}"
                )
                model.save_networks('best_psnr')

            # Save best model based on SSIM
            if current_ssim > best_ssim:
                best_ssim = current_ssim
                best_ssim_psnr = current_psnr
                best_ssim_std = current_ssim_std
                best_psnr_std_for_ssim = current_psnr_std
                best_ssim_epoch = epoch
                logger.info(
                    f"New best SSIM model found! "
                    f"SSIM: {best_ssim:.4f}±{best_ssim_std:.4f}, "
                    f"PSNR: {best_ssim_psnr:.2f}±{best_psnr_std_for_ssim:.2f} "
                    f"at epoch {best_ssim_epoch}"
                )
                model.save_networks('best_ssim')

            # Save best model based on combined score
            if combined_score > best_combined:
                best_combined = combined_score
                best_combined_psnr = current_psnr
                best_combined_ssim = current_ssim
                best_combined_psnr_std = current_psnr_std
                best_combined_ssim_std = current_ssim_std
                best_combined_epoch = epoch
                logger.info(
                    f"New best combined model found! Combined Score: {best_combined:.4f} "
                    f"(PSNR: {best_combined_psnr:.2f}±{best_combined_psnr_std:.2f}, "
                    f"SSIM: {best_combined_ssim:.4f}±{best_combined_ssim_std:.4f}) "
                    f"at epoch {best_combined_epoch}"
                )
                model.save_networks('best_combined')

            # Print current best results after each epoch
            logger.info(f"Current best results after epoch {epoch}:")
            logger.info(
                f"  Best PSNR: {best_psnr:.2f}±{best_psnr_std:.2f} "
                f"(SSIM: {best_psnr_ssim:.4f}±{best_ssim_std_for_psnr:.4f}) "
                f"at epoch {best_psnr_epoch}"
            )
            logger.info(
                f"  Best SSIM: {best_ssim:.4f}±{best_ssim_std:.4f} "
                f"(PSNR: {best_ssim_psnr:.2f}±{best_psnr_std_for_ssim:.2f}) "
                f"at epoch {best_ssim_epoch}"
            )
            logger.info(
                f"  Best Combined Score: {best_combined:.4f} "
                f"(PSNR: {best_combined_psnr:.2f}±{best_combined_psnr_std:.2f}, "
                f"SSIM: {best_combined_ssim:.4f}±{best_combined_ssim_std:.4f}) "
                f"at epoch {best_combined_epoch}"
            )

        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        logger.info(
            f"End of epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay} \t Time taken: {epoch_time:.2f} seconds"
        )
        model.update_learning_rate()