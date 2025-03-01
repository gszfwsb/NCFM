import torch
import numpy as np
import os
import time
import matplotlib

matplotlib.use("Agg")
__all__ = ["Compose", "Lighting", "ColorJitter"]
import matplotlib.pyplot as plt


class TimingTracker:
    def __init__(self, logger):
        self.print = logger
        self.timing_stats = {"data": 0, "aug": 0, "loss": 0, "backward": 0}

    def start_step(self):
        self.step_start_time = time.time()

    def record(self, phase):
        current_time = time.time()
        self.timing_stats[phase] += current_time - self.step_start_time
        self.step_start_time = current_time

    def report(self, reset=True):
        total_time = sum(self.timing_stats.values())
        summary = ", ".join(
            f"{key}:{value:.2f}s({value / total_time * 100:.1f}%)"
            for key, value in self.timing_stats.items()
        )
        if reset:
            self.reset_stats()
        return summary

    def reset_stats(self):
        self.timing_stats = {key: 0 for key in self.timing_stats}


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger:
    def __init__(self, path):
        self.logger = open(os.path.abspath(os.path.join(path, "print.log")), "w")

    def __call__(self, string, end="\n", print_=True):
        if print_:
            print("{}".format(string), end=end)
            if end == "\n":
                self.logger.write("{}\n".format(string))
            else:
                self.logger.write("{} ".format(string))
            self.logger.flush()


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


class LossPlotter:
    def __init__(
        self,
        save_path,
        filename_pattern,
        dataset,
        ipc,
        dis_metrics,
        optimizer_info,
        ncfd_distribution="gussian",
    ):
        """
        Initializes the LossPlotter with paths, dataset details, and optimizer settings.
        """
        self.save_path = save_path
        self.filename_pattern = filename_pattern
        self.dataset = dataset
        self.ipc = ipc
        self.dis_metrics = dis_metrics
        self.ncfd_distribution = ncfd_distribution
        self.optimizer_info = optimizer_info

        # Initialize tracking lists for sigma values and loss/accuracy data
        self.sigma_history = []
        self.loss_match_data = []
        self.loss_calib_data = []
        self.acc_data = {}

        # Create the save directory if it doesn't exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _get_optimizer_str(self):
        """Generates a string representing the optimizer information."""
        opt_type = self.optimizer_info["type"].upper()
        lr = self.optimizer_info["lr"]
        if opt_type in ["ADAM", "ADAMW"]:
            return f"{opt_type}(lr={lr:.4f}, wd={self.optimizer_info['weight_decay']})"
        return f"{opt_type}(lr={lr:.4f})"

    def update_sigma(self, sigma):
        """
        Updates the sigma history with the new sigma value.

        Parameters:
        sigma : np.ndarray or torch.Tensor
            The sigma value for the current iteration.
        """
        self.sigma_history.append(sigma)

    def update_match_loss(self, loss):
        """
        Updates the match loss data.

        Parameters:
        loss : torch.Tensor
            The loss value for the current iteration.
        """
        self.loss_match_data.append(loss)

    def update_calib_loss(self, loss):
        """
        Updates the calibration loss data.

        Parameters:
        loss : torch.Tensor
            The calibration loss value for the current iteration.
        """
        self.loss_calib_data.append(loss)

    def plot_and_save_loss_curve(self):
        """
        Plots and saves the loss and accuracy trends.
        """
        # Check if there is any data to plot
        has_loss_data = len(self.loss_match_data) > 0
        has_calib_data = len(self.loss_calib_data) > 0
        has_acc_data = len(self.acc_data) > 0

        if not has_loss_data and not has_acc_data and not has_calib_data:
            print("No loss or accuracy data to plot.")
            return

        # Create a figure and axis for plotting
        fig, ax1 = plt.subplots(figsize=(8, 5))

        # Plot the match loss if available
        if has_loss_data:
            color = "tab:red"
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Loss (Match)", color=color)
            ax1.plot(
                range(len(self.loss_match_data)),
                self.loss_match_data,
                linestyle="-",
                color=color,
            )
            ax1.tick_params(axis="y", labelcolor=color)

        # Plot the calibration loss if available
        if has_calib_data:
            color = "tab:green"
            if has_loss_data:
                # If match loss is plotted, use a second y-axis
                ax2 = ax1.twinx()
                ax2.set_ylabel("Loss (Calib)", color=color)
                ax2.plot(
                    range(len(self.loss_calib_data)),
                    self.loss_calib_data,
                    linestyle="-",
                    color=color,
                )
                ax2.tick_params(axis="y", labelcolor=color)
            else:
                # If no match loss, plot calibration loss on the first axis
                ax1.set_ylabel("Loss (Calib)", color=color)
                ax1.plot(
                    range(len(self.loss_calib_data)),
                    self.loss_calib_data,
                    linestyle="-",
                    color=color,
                )
                ax1.tick_params(axis="y", labelcolor=color)

        # Plot the accuracy if available
        if has_acc_data:
            iters = sorted(self.acc_data.keys())
            acc_values = [self.acc_data[it] for it in iters]

            if has_loss_data or has_calib_data:
                # Create a second y-axis for accuracy if loss is also plotted
                ax2 = ax1.twinx()
                color = "tab:blue"
                ax2.set_ylabel("Validation Mean Accuracy", color=color)
                ax2.plot(iters, acc_values, linestyle="--", color=color)
                ax2.tick_params(axis="y", labelcolor=color)
            else:
                # If no loss data, plot accuracy on the first axis
                color = "tab:blue"
                ax1.set_ylabel("Validation Mean Accuracy", color=color)
                ax1.plot(iters, acc_values, linestyle="--", color=color)
                ax1.tick_params(axis="y", labelcolor=color)

        # Set the title of the plot with dataset and optimizer information
        plt.title(
            f"{self.dataset} - IPC {self.ipc} - {self.dis_metrics}\n"
            f"{self.ncfd_distribution.capitalize()} - {self._get_optimizer_str()}"
        )

        fig.tight_layout()

        # Save the plot as a PNG file
        file_name = os.path.join(
            self.save_path, f"{self.filename_pattern}_loss_acc.png"
        )
        plt.savefig(file_name)
        plt.close()
