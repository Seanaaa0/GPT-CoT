import os
import json
import argparse
import matplotlib.pyplot as plt


def plot_trainer_state(json_path, metrics="123"):
    # 讀取 JSON 資料
    with open(json_path, "r") as f:
        data = json.load(f)

    # 過濾出對應的資料點（排除 None）
    train_points = [(x["step"], x["loss"]) for x in data["log_history"]
                    if "loss" in x and x["loss"] is not None]
    eval_points = [(x["step"], x["eval_loss"]) for x in data["log_history"]
                   if "eval_loss" in x and x["eval_loss"] is not None]
    grad_points = [(x["step"], x["grad_norm"]) for x in data["log_history"]
                   if "grad_norm" in x and x["grad_norm"] is not None]
    lr_points = [(x["step"], x["learning_rate"]) for x in data["log_history"]
                 if "learning_rate" in x and x["learning_rate"] is not None]

    # 解包資料
    steps_train, train_loss = zip(*train_points) if train_points else ([], [])
    steps_eval, eval_loss = zip(*eval_points) if eval_points else ([], [])
    steps_grad, grad_norm = zip(*grad_points) if grad_points else ([], [])
    steps_lr, learning_rate = zip(*lr_points) if lr_points else ([], [])

    # 開始繪圖
    fig_id = 1
    plt.figure(figsize=(8, 12))

    if "1" in metrics:
        plt.subplot(3, 1, fig_id)
        plt.plot(steps_train, train_loss, label="Training Loss", color="blue")
        plt.plot(steps_eval, eval_loss, label="Eval Loss", color="orange")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.legend()
        fig_id += 1

    if "2" in metrics:
        plt.subplot(3, 1, fig_id)
        plt.plot(steps_grad, grad_norm, color="green")
        plt.xlabel("Step")
        plt.ylabel("Norm")
        plt.title("Gradient Norm")
        fig_id += 1

    if "3" in metrics:
        plt.subplot(3, 1, fig_id)
        plt.plot(steps_lr, learning_rate, color="orange")
        plt.xlabel("Step")
        plt.ylabel("LR")
        plt.title("Learning Rate")

    # 儲存圖片到 png/資料夾
    filename_without_ext = os.path.splitext(os.path.basename(json_path))[0]
    save_dir = os.path.join(os.path.dirname(json_path), "..", "png")
    os.makedirs(save_dir, exist_ok=True)
    image_filename = os.path.join(save_dir, f"{filename_without_ext}.png")

    plt.tight_layout()
    plt.savefig(image_filename)
    print(f"✅ 圖片已儲存為 {image_filename}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Trainer State Metrics")
    parser.add_argument(
        "--file", type=str, default="trainer_state.json",
        help="Path to trainer_state.json (ex: trainer_state/xxx.json)"
    )
    parser.add_argument(
        "--metrics", type=str, default="1",
        help="Select metrics to plot: 1=Loss, 2=Grad, 3=LR. E.g., 123 or 12"
    )

    args = parser.parse_args()
    plot_trainer_state(args.file, args.metrics)
