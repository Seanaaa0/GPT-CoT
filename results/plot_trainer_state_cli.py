import json
import argparse
import matplotlib.pyplot as plt
import os


def plot_trainer_state(json_path: str, options: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps_all = []
    losses_all = []
    eval_losses_all = []
    grad_norms_all = []
    learning_rates_all = []

    for entry in data["log_history"]:
        if "step" in entry:
            steps_all.append(entry["step"])
            losses_all.append(entry.get("loss", None))
            eval_losses_all.append(entry.get("eval_loss", None))
            grad_norms_all.append(entry.get("grad_norm", None))
            learning_rates_all.append(entry.get("learning_rate", None))

    # 判斷要畫哪些圖
    show_loss = '1' in options
    show_grad = '2' in options
    show_lr = '3' in options

    total_plots = show_loss + show_grad + show_lr
    if total_plots == 0:
        print("⚠️ 你沒有選任何圖！請輸入 1（Loss）、2（Grad）、3（LR）之一，例如: 12 或 123")
        return

    fig, axs = plt.subplots(total_plots, 1, figsize=(10, 4 * total_plots))
    if total_plots == 1:
        axs = [axs]

    plot_idx = 0

    if show_loss:
        # 過濾 None 值
        train_steps = [s for s, l in zip(
            steps_all, losses_all) if l is not None]
        train_losses = [l for l in losses_all if l is not None]
        eval_steps = [s for s, l in zip(
            steps_all, eval_losses_all) if l is not None]
        eval_losses = [l for l in eval_losses_all if l is not None]

        axs[plot_idx].plot(train_steps, train_losses, label="Training Loss")
        axs[plot_idx].plot(eval_steps, eval_losses, label="Eval Loss")
        axs[plot_idx].set_title("Loss")
        axs[plot_idx].set_xlabel("Step")
        axs[plot_idx].set_ylabel("Loss")
        axs[plot_idx].legend()
        axs[plot_idx].grid(True)
        plot_idx += 1

    if show_grad:
        grad_steps = [s for s, g in zip(
            steps_all, grad_norms_all) if g is not None]
        grad_norms = [g for g in grad_norms_all if g is not None]

        axs[plot_idx].plot(grad_steps, grad_norms,
                           label="Gradient Norm", color="green")
        axs[plot_idx].set_title("Gradient Norm")
        axs[plot_idx].set_xlabel("Step")
        axs[plot_idx].set_ylabel("Norm")
        axs[plot_idx].grid(True)
        plot_idx += 1

    if show_lr:
        lr_steps = [s for s, lr in zip(
            steps_all, learning_rates_all) if lr is not None]
        lrs = [lr for lr in learning_rates_all if lr is not None]

        axs[plot_idx].plot(
            lr_steps, lrs, label="Learning Rate", color="orange")
        axs[plot_idx].set_title("Learning Rate")
        axs[plot_idx].set_xlabel("Step")
        axs[plot_idx].set_ylabel("LR")
        axs[plot_idx].grid(True)

    plt.tight_layout()

    # 儲存圖片
    filename_without_ext = os.path.splitext(os.path.basename(json_path))[0]
    image_filename = f"{filename_without_ext}.png"
    plt.savefig(image_filename)
    print(f"✅ 圖片已儲存為 {image_filename}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Trainer State Metrics")
    parser.add_argument(
        "--file", type=str, default="trainer_state.json", help="Path to trainer_state.json")
    parser.add_argument("--metrics", type=str, default="1",
                        help="Select metrics to plot: 1=Loss, 2=Grad, 3=LR. E.g., 123 or 12")

    args = parser.parse_args()
    plot_trainer_state(args.file, args.metrics)
