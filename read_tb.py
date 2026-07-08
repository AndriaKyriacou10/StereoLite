import tensorboard as tb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import argparse
def main():
    if args.phase == 1:
        log_dir = './runs/phase1_training_Jun26_13-52-53'
        tag_name = 'Validation Summary/text_summary'
    else:
        log_dir = './runs/phase2_training_Jun27_00-50-50'
        tag_name = 'Phase 2: Validation Summary/text_summary'
        
    
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
    plt.title(f'Validation EPE Phase {args.phase}')
    plt.grid(visible=True)
    plt.savefig(f'validation_epe_phase{args.phase}.png', dpi=400)
    plt.close()

    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=2, help='Phase of training (1 or 2)')
    args = parser.parse_args()
    main()
