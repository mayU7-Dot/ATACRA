"""Combined v3 queue: all remaining retrain + extra-seed tasks in ONE launcher
(single GPU reservation pool avoids cross-launcher races).
"""
import subprocess, os, time
from concurrent.futures import ThreadPoolExecutor
import threading

BASE = '/home/raid-5/MYQ/pneumonia-miccai2024'
SEEDS = [42, 3407, 2024, 2025, 2026]
EXTRA_SEEDS = [1234, 777, 888, 999, 31415]
SPLIT_BIN = f'{BASE}/data_splits/split_disjoint_70_15_15.json'
SPLIT_TRI = f'{BASE}/data_splits/split_3class.json'
WARMSTART = f'{BASE}/checkpoints/warmstart_clean_v2.pth'
DATA_BIN = '/home/raid-5/MYQ/chest_xray'
DATA_TRI = '/home/raid-5/MYQ/Data'

MODELS = {
    'hybrid_v3':                    ('bin', ['--arch', 'hybrid']),
    'atacra_v3_warmup3':            ('bin', ['--arch', 'hybrid_atacra', '--warmup_epochs', '3',
                                             '--acra_lr_mult', '100', '--alpha_init', '0.1',
                                             '--prior_ipsi', '0.2', '--prior_contra', '0.1']),
    'zeroqkv_fixed_v3':             ('bin', ['--arch', 'hybrid_acra', '--fixed_alpha',
                                             '--acra_lr_mult', '100', '--alpha_init', '0.1']),
    'acra_v3':                      ('bin', ['--arch', 'hybrid_acra',
                                             '--acra_lr_mult', '100', '--alpha_init', '0.1']),
    'atacra_nodelta_v3':            ('bin', ['--arch', 'hybrid_atacra', '--freeze_topo',
                                             '--acra_lr_mult', '100', '--alpha_init', '0.1',
                                             '--prior_ipsi', '0.2', '--prior_contra', '0.1']),
    'atacra_noprior_v3':            ('bin', ['--arch', 'hybrid_atacra', '--no_prior',
                                             '--acra_lr_mult', '100', '--alpha_init', '0.1',
                                             '--prior_ipsi', '0.2', '--prior_contra', '0.1']),
    'stdqkv_atacra_v3':             ('bin', ['--arch', 'hybrid_stdqkv',
                                             '--acra_lr_mult', '100', '--alpha_init', '0.1',
                                             '--prior_ipsi', '0.2', '--prior_contra', '0.1']),
    'atacra_randprior_v3':          ('bin', ['--arch', 'hybrid_atacra', '--random_prior',
                                             '--shuffle_seed', '1234',
                                             '--acra_lr_mult', '100', '--alpha_init', '0.1',
                                             '--prior_ipsi', '0.2', '--prior_contra', '0.1']),
    'atacra_randprior_nodelta_v3':  ('bin', ['--arch', 'hybrid_atacra', '--random_prior',
                                             '--shuffle_seed', '1234', '--freeze_topo',
                                             '--acra_lr_mult', '100', '--alpha_init', '0.1',
                                             '--prior_ipsi', '0.2', '--prior_contra', '0.1']),
    'atacra_v3':                    ('bin', ['--arch', 'hybrid_atacra',
                                             '--acra_lr_mult', '100', '--alpha_init', '0.1',
                                             '--prior_ipsi', '0.2', '--prior_contra', '0.1']),
    'hybrid_3class_v3':             ('tri', ['--arch', 'hybrid', '--num_classes', '3']),
    'atacra_3class_v3_warmup3':     ('tri', ['--arch', 'hybrid_atacra', '--num_classes', '3',
                                             '--warmup_epochs', '3',
                                             '--acra_lr_mult', '100', '--alpha_init', '0.1',
                                             '--prior_ipsi', '0.2', '--prior_contra', '0.1']),
}
EXTRA_MODELS = ['hybrid_v3', 'atacra_v3_warmup3']

gpu_lock = threading.Lock()
gpus_in_use = set()


def finished(model, seed):
    logfile = f'{BASE}/results/{model}/seed{seed}.txt'
    if os.path.exists(logfile) and 'Final Test-Set' in open(logfile).read():
        return True
    pat = f'train_binary[.]py.*{model}/ckpt_seed{seed}'
    r = subprocess.run(['pgrep', '-f', pat], capture_output=True)
    return r.returncode == 0


def acquire_gpu():
    while True:
        with gpu_lock:
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader'],
                capture_output=True, text=True).stdout
            for line in out.strip().splitlines():
                idx_s, used_s = line.split(',')
                idx = int(idx_s)
                used = int(used_s.split()[0])
                if used < 1024 and idx not in gpus_in_use:
                    gpus_in_use.add(idx)
                    return idx
        time.sleep(20)


def release_gpu(gpu):
    with gpu_lock:
        gpus_in_use.discard(gpu)


def run_one(model, seed):
    gpu = acquire_gpu()
    try:
        task, flags = MODELS[model]
        outdir = f'{BASE}/results/{model}'
        os.makedirs(outdir, exist_ok=True)
        logfile = f'{outdir}/seed{seed}.txt'
        if task == 'tri':
            data, split = DATA_TRI, SPLIT_TRI
        else:
            data, split = DATA_BIN, SPLIT_BIN
        cmd = ['python', f'{BASE}/train_binary.py',
               '--data_path', data,
               '--backbone', 'resnet50', '--image_size', '512',
               '--batch_size', '16', '--epochs', '30',
               '--lr', '1e-5', '--weight_decay', '1e-4',
               '--num_workers', '4', '--val_split', '0.2',
               '--tta',
               '--seed', str(seed), '--split_file', split,
               '--no_balance', '--class_weights', '--warmstart', WARMSTART,
               ] + flags
        if task == 'bin' and '--warmup_epochs' not in flags:
            cmd += ['--label_smoothing', '0.1']
        cmd += ['--save_dir', f'{BASE}/results/{model}/ckpt_seed{seed}/']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        with open(logfile, 'w') as f:
            subprocess.run(cmd, cwd=BASE, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(f'[done] {model} seed{seed} on GPU {gpu}', flush=True)
    finally:
        release_gpu(gpu)


def main():
    tasks = [(m, s) for m in MODELS for s in SEEDS if not finished(m, s)]
    tasks += [(m, s) for m in EXTRA_MODELS for s in EXTRA_SEEDS if not finished(m, s)]
    print(f'{len(tasks)} remaining tasks', flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda t: run_one(*t), tasks))
    print('=== ALL COMBINED V3 TASKS DONE ===', flush=True)


if __name__ == '__main__':
    main()
