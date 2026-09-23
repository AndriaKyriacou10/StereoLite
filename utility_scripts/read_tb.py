import tensorboard as tb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import argparse
import matplotlib.ticker as mticker

def plot_validation(log_dir, tag_name, run_id = 1):
    event_acc = event_accumulator.EventAccumulator(log_dir, size_guidance={'tensors':0})

    event_acc.Reload()
    
    epe_np = None

    
    if tag_name in event_acc.Tags()['tensors']:
        tensor_events = event_acc.Tensors(tag_name)
        epe_np = np.ones(shape=(len(tensor_events),), dtype=np.float32)
        for idx, event in enumerate(tensor_events):
            # epe_str = None
            epe_str = event.tensor_proto.string_val[0].decode('utf-8').split('=')[1].strip()
            epe = np.float32(epe_str)
            epe_np[idx] = epe
        
    plt.figure()
    plt.plot(epe_np, linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('EPE')
    plt.title(f'Validation EPE Phase {args.phase} RUN {run_id}')
    plt.grid(visible=True)
    plt.savefig(f'validation_epe_phase{args.phase}_RUN{run_id}.png', dpi=400)
    plt.close()

    print('Done')

def read_tensorboard(log_dir, tag_name):
    event_acc = event_accumulator.EventAccumulator(log_dir, size_guidance={'tensors':0})
    event_acc.Reload()
    print(event_acc.Tags()['tensors'])
    tensor_events = event_acc.Tensors(tag_name)
    best_epe_cv = {'middlebury': (float('inf'), -1), 'eth3d': (float('inf'), 1), 'sceneflow': (float('inf'), 1)}
    best_epe_no_cv = {'middlebury': (float('inf'), -1), 'eth3d': (float('inf'), 1), 'sceneflow': (float('inf'), 1)}
    
    mid_data, eth_data, sf_data = [], [], []
    for idx, event in enumerate(tensor_events):
        val_str = event.tensor_proto.string_val[0].decode('utf-8')
        epe = np.float32(val_str.split('=')[1].strip())
        if 'middlebury' in val_str:
            mid_data.append((event.step, epe))
            if epe < best_epe_cv['middlebury'][0] and idx < 90000:
                best_epe_cv['middlebury'] = (epe, event.step)
            elif epe < best_epe_no_cv['middlebury'][0]:
                best_epe_no_cv['middlebury'] = (epe, event.step)
        elif 'eth3d' in val_str:
            eth_data.append((event.step, epe))
            if epe < best_epe_cv['eth3d'][0] and idx < 90000:
                best_epe_cv['eth3d'] = (epe, event.step)
            elif epe < best_epe_no_cv['eth3d'][0]:
                best_epe_no_cv['eth3d'] = (epe, event.step)
        elif 'sceneflow' in val_str:
            sf_data.append((event.step, epe))
            if epe < best_epe_cv['sceneflow'][0] and idx < 90000:
                best_epe_cv['sceneflow'] = (epe, event.step)
            elif epe < best_epe_no_cv['sceneflow'][0]:
                best_epe_no_cv['sceneflow'] = (epe, event.step)
            
    
    # print(f"Best EPE for Middlebury with CV: {best_epe_cv['middlebury'][0]} at index {best_epe_cv['middlebury'][1]}")
    # print(f"Best EPE for ETH3D with CV: {best_epe_cv['eth3d'][0]} at index {best_epe_cv['eth3d'][1]}")
    # print(f"Best EPE for SceneFlow with CV: {best_epe_cv['sceneflow'][0]} at index {best_epe_cv['sceneflow'][1]}")
    # print("--------------------------------------------------")
    # print(f"Best EPE for Middlebury without CV: {best_epe_no_cv['middlebury'][0]} at index {best_epe_no_cv['middlebury'][1]}")
    # print(f"Best EPE for ETH3D without CV: {best_epe_no_cv['eth3d'][0]} at index {best_epe_no_cv['eth3d'][1]}")
    # print(f"Best EPE for SceneFlow without CV: {best_epe_no_cv['sceneflow'][0]} at index {best_epe_no_cv['sceneflow'][1]}")
    
    return mid_data, eth_data, sf_data

def plot_series(ax, data_l1, data_l2, label):
    if len(data_l1) == 0 or len(data_l2) == 0:
        return
    data_l1 = sorted(data_l1)  # guard against out-of-order events
    steps, epes_l1 = zip(*data_l1)
    data_l2 = sorted(data_l2)  # guard against out-of-order events
    steps, epes_l2 = zip(*data_l2)

    ax.plot(steps, epes_l1, label="Only Layer1", linewidth=1.5, color="#032FB2")
    ax.plot(steps, epes_l2, label="Layer1 + Layer2", linewidth=1.5, color="#17AB37")
    ax.vlines(x=90000, ymin=-0.2, ymax=np.max(epes_l1), linestyle='dashed', color='red', label='CV Disabled')

    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel('Validation EPE', fontsize=12)
    ax.grid(True)
    ax.set_title(f'{label}', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    
    # Limit number of x-axis ticks and format as "Nk"
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=2, help='Phase of training (1 or 2)')
    args = parser.parse_args()
    
    log_dir_layer1 = './runs/train_continuous_two_cycle_Aug31_15-29-59_cutoff_90k_layer1'
    log_dir_layer2 = './runs/train_continuous_two_cycle_Aug30_03-08-40_cutoff_90k'
    tag_name = 'Validation - Dataset/text_summary'
    # plot_validation(log_dir, tag_name, run_id=2)
    
    mid_data_layer1, eth_data_layer1, sf_data_layer1 = read_tensorboard(log_dir_layer1, 'Validation - Dataset/text_summary')
    mid_data_layer2, eth_data_layer2, sf_data_layer2 = read_tensorboard(log_dir_layer2, 'Validation - Dataset/text_summary')
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    plot_series(axes[0], mid_data_layer1, mid_data_layer2, label='Middlebury')
    plot_series(axes[1], eth_data_layer1, eth_data_layer2, label='ETH3D')
    plot_series(axes[2], sf_data_layer1, sf_data_layer2, label='SceneFlow')
    
    fig.suptitle('Validation EPE vs ContextNet Capacity', fontsize=16)
    plt.tight_layout()

    plt.savefig('val_epe_VS_contextNet_capacity_all.eps', dpi=500, bbox_inches='tight')
    plt.savefig('val_epe_VS_contextNet_capacity_all.png', dpi=500, bbox_inches='tight')
    
    # plt.grid(visible=True)
    # plt.xlabel('Steps')
    # plt.ylabel('Val. EPE')
    # plt.title("Validation EPE across Datasets")
    # plt.vlines(x=90000, ymin=0, ymax=100, colors='red', linestyles='dashed', label='Cutoff at 90k steps')
    # plt.legend()
    # plt.savefig(f"train_continuous_val_epe.png", dpi=400)