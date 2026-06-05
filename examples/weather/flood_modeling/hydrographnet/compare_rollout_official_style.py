import argparse
import csv
import logging
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import torch

from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from physicsnemo.utils import load_checkpoint
from torch_geometric.utils import to_networkx


logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def create_animation(
    rollout_predictions,
    ground_truth,
    initial_graph,
    rmse_list,
    output_path,
    time_per_step=20 / 60,
):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20

    fig, axes = plt.subplots(2, 2, figsize=(30, 30))
    cax1 = fig.add_axes([0.05, 0.53, 0.02, 0.35])
    cax2 = fig.add_axes([0.95, 0.53, 0.02, 0.35])
    cax3 = fig.add_axes([0.05, 0.1, 0.02, 0.35])

    init_node_feats = initial_graph.x
    pos = {
        i: (init_node_feats[i, 0].item(), init_node_feats[i, 1].item())
        for i in range(init_node_feats.shape[0])
    }
    graph_nx = to_networkx(initial_graph.cpu()).to_undirected()

    all_vals = torch.cat(rollout_predictions + ground_truth)
    vmin_global = all_vals.min().item()
    vmax_global = all_vals.max().item()

    def update(frame):
        for ax in axes.flat:
            ax.clear()
        current_time = (frame + 1) * time_per_step

        pred_vals = rollout_predictions[frame].cpu().numpy()
        nodes_pred = nx.draw_networkx_nodes(
            graph_nx,
            pos,
            node_color=pred_vals,
            node_size=250,
            cmap=plt.cm.viridis,
            ax=axes[0, 0],
            vmin=vmin_global,
            vmax=vmax_global,
            node_shape="s",
        )
        nx.draw_networkx_edges(graph_nx, pos, alpha=0.5, ax=axes[0, 0])
        axes[0, 0].set_title(f"Time {current_time:.2f} Hours - Prediction", fontsize=24)
        fig.colorbar(nodes_pred, cax=cax1)

        gt_vals = ground_truth[frame].cpu().numpy()
        nodes_gt = nx.draw_networkx_nodes(
            graph_nx,
            pos,
            node_color=gt_vals,
            node_size=250,
            cmap=plt.cm.viridis,
            ax=axes[0, 1],
            vmin=vmin_global,
            vmax=vmax_global,
            node_shape="s",
        )
        nx.draw_networkx_edges(graph_nx, pos, alpha=0.5, ax=axes[0, 1])
        axes[0, 1].set_title(
            f"Time {current_time:.2f} Hours - Ground Truth", fontsize=24
        )
        fig.colorbar(nodes_gt, cax=cax2)

        abs_vals = torch.abs(rollout_predictions[frame] - ground_truth[frame]).numpy()
        nodes_error = nx.draw_networkx_nodes(
            graph_nx,
            pos,
            node_color=abs_vals,
            node_size=250,
            cmap=plt.cm.viridis,
            ax=axes[1, 0],
            vmin=vmin_global,
            vmax=vmax_global,
            node_shape="s",
        )
        nx.draw_networkx_edges(graph_nx, pos, alpha=0.5, ax=axes[1, 0])
        axes[1, 0].set_title(
            f"Time {current_time:.2f} Hours - Absolute Error", fontsize=24
        )
        fig.colorbar(nodes_error, cax=cax3)

        times = [(i + 1) * time_per_step for i in range(frame + 1)]
        axes[1, 1].plot(
            times,
            rmse_list[: frame + 1],
            label="Water Depth RMSE",
            color="b",
            linewidth=3,
        )
        axes[1, 1].set_title("RMSE Over Time", fontsize=24)
        axes[1, 1].set_xlabel("Time (Hours)", fontsize=24)
        axes[1, 1].set_ylabel("RMSE", fontsize=24)
        axes[1, 1].legend(fontsize=20)
        axes[1, 1].grid(True)

    ani = animation.FuncAnimation(fig, update, frames=len(rollout_predictions), repeat=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(output_path, writer="pillow", fps=2)
    plt.close(fig)
    print(f"Animation saved to {output_path}")


def run_rollout(model, graph, rollout_data, rollout_length, n_time_steps, device):
    graph = graph.to(device)
    edge_features = graph.edge_attr.to(device)
    x_iter = graph.x.to(device)
    num_nodes = x_iter.size(0)

    inflow_seq = rollout_data["inflow"].to(device)
    precip_seq = rollout_data["precipitation"].to(device)
    wd_gt_seq = rollout_data["water_depth_gt"].to(device)

    rollout_preds = []
    ground_truth = []
    step_rmse = []

    with torch.no_grad():
        for t in range(rollout_length):
            static_part = x_iter[:, :12]
            water_depth_window = x_iter[:, 12 : 12 + n_time_steps]
            volume_window = x_iter[:, 12 + n_time_steps : 12 + 2 * n_time_steps]
            x_input = torch.cat([static_part, water_depth_window, volume_window], dim=1)

            pred = model(x_input, edge_features, graph)
            new_wd = water_depth_window[:, -1:] + pred[:, 0:1]
            new_vol = volume_window[:, -1:] + pred[:, 1:2]

            water_depth_updated = torch.cat([water_depth_window[:, 1:], new_wd], dim=1)
            volume_updated = torch.cat([volume_window[:, 1:], new_vol], dim=1)
            static_part_updated = static_part.clone()
            new_flow = inflow_seq[t].unsqueeze(0).expand(num_nodes, 1)
            new_precip = precip_seq[t].unsqueeze(0).expand(num_nodes, 1)
            static_part_updated[:, 10:12] = torch.cat([new_flow, new_precip], dim=1)
            x_iter = torch.cat(
                [static_part_updated, water_depth_updated, volume_updated], dim=1
            )

            pred_wd = new_wd.squeeze(1).detach().cpu()
            gt_wd = wd_gt_seq[t].detach().cpu()
            rmse = torch.sqrt(torch.mean((new_wd.squeeze(1) - wd_gt_seq[t]) ** 2))
            rollout_preds.append(pred_wd)
            ground_truth.append(gt_wd)
            step_rmse.append(float(rmse.detach().cpu()))

    return rollout_preds, ground_truth, step_rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--test-ids-file", default="test_h13h15.txt")
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--rollout-length", type=int, default=25)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--animation-dir")
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--n-time-steps", type=int, default=2)
    parser.add_argument("--num-input-features", type=int, default=16)
    parser.add_argument("--num-edge-features", type=int, default=3)
    parser.add_argument("--num-output-features", type=int, default=2)
    parser.add_argument("--skip-animations", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HydroGraphDataset(
        data_dir=args.data_dir,
        prefix=args.prefix,
        n_time_steps=args.n_time_steps,
        hydrograph_ids_file=args.test_ids_file,
        split="test",
        rollout_length=args.rollout_length,
        return_physics=False,
    )
    model = MeshGraphKAN(
        args.num_input_features,
        args.num_edge_features,
        args.num_output_features,
    ).to(device)
    epoch_loaded = load_checkpoint(
        args.ckpt_path,
        models=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        device=device,
    )
    model.eval()

    rows = []
    all_step_rmse = []
    for idx in range(len(dataset)):
        graph, rollout_data = dataset[idx]
        rollout_preds, ground_truth, step_rmse = run_rollout(
            model, graph, rollout_data, args.rollout_length, args.n_time_steps, device
        )
        hydro_id = dataset.dynamic_data[idx]["hydro_id"]
        all_step_rmse.append(step_rmse)
        rows.append(
            {
                "hydrograph_id": hydro_id,
                "epoch_loaded": epoch_loaded,
                "rollout_length": args.rollout_length,
                "mean_rmse": sum(step_rmse) / len(step_rmse),
                "final_step_rmse": step_rmse[-1],
                **{f"rmse_step_{i + 1}": value for i, value in enumerate(step_rmse)},
            }
        )
        print(f"Hydrograph {hydro_id}: Mean RMSE = {rows[-1]['mean_rmse']:.4f}")
        if args.animation_dir and not args.skip_animations:
            create_animation(
                rollout_preds,
                ground_truth,
                graph.cpu(),
                step_rmse,
                Path(args.animation_dir) / f"animation_{hydro_id}.gif",
            )

    step_tensor = torch.tensor(all_step_rmse)
    rows.append(
        {
            "hydrograph_id": "OVERALL_MEAN",
            "epoch_loaded": epoch_loaded,
            "rollout_length": args.rollout_length,
            "mean_rmse": float(step_tensor.mean()),
            "final_step_rmse": float(step_tensor[:, -1].mean()),
            **{
                f"rmse_step_{i + 1}": float(value)
                for i, value in enumerate(step_tensor.mean(dim=0))
            },
        }
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
