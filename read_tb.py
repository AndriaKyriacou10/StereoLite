import tensorboard as tb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import argparse

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
    best_epe = {'middlebury': (float('inf'), -1), 'eth3d': (float('inf'), 1), 'sceneflow': (float('inf'), 1)}
    for idx, event in enumerate(tensor_events):
        val_str = event.tensor_proto.string_val[0].decode('utf-8')
        epe = np.float32(val_str.split('=')[1].strip())
        if 'middlebury' in val_str:
            if epe < best_epe['middlebury'][0]:
                best_epe['middlebury'] = (epe, event.step)
        elif 'eth3d' in val_str:
            if epe < best_epe['eth3d'][0]:
                best_epe['eth3d'] = (epe, event.step)
        elif 'sceneflow' in val_str:
            if epe < best_epe['sceneflow'][0]:
                best_epe['sceneflow'] = (epe, event.step)
            
    
    print(f"Best EPE for Middlebury: {best_epe['middlebury'][0]} at index {best_epe['middlebury'][1]}")
    print(f"Best EPE for ETH3D: {best_epe['eth3d'][0]} at index {best_epe['eth3d'][1]}")
    print(f"Best EPE for SceneFlow: {best_epe['sceneflow'][0]} at index {best_epe['sceneflow'][1]}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=2, help='Phase of training (1 or 2)')
    args = parser.parse_args()
    
    if args.phase == 1:
        log_dir = './runs/phase1_training_Jul16_19-18-08'
        tag_name = 'Validation Summary/text_summary'
    else:
        log_dir = './runs/phase2_training_Jun27_00-50-50'
        tag_name = 'Phase 2: Validation Summary/text_summary'
        
    plot_validation(log_dir, tag_name, run_id=2)
    read_tensorboard(log_dir, 'Validation - Dataset/text_summary')
