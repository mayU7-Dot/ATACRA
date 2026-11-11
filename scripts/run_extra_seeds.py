"""Extra seeds for the v3 main-table pair (hybrid_v3, atacra_v3_warmup3).

Interleaves with the main v3 retrain via dynamic GPU acquisition.
Extra seeds: {1234, 777, 888, 999, 31415} — combined with the base 5 seeds
gives n=10 per model for the paired comparison.
"""
import subprocess, os, time
from concurrent.futures import ThreadPoolExecutor
import threading

BASE = '/home/raid-5/MYQ/pneumonia-miccai2024'
EXTRA_SEEDS = [1234, 777, 888, 999, 31415]
SPLIT = f'{BASE}/data_splits/split_disjoint_70_15_15.json'
WARMSTART = f'{BASE}/checkpoints/warmstart_clean_v2.pth'

MODELS = {
    'hybrid_v3': ['--arch', 'hybrid', '--label_smoothing', '0.1'],
    'atacra_v3_warmup3': ['--arch', 'hybrid_atacra', '--warmup_epochs', '3',
                          '--acra_lr_mult', '100', '--alpha_init', '0.1',
                          '--prior_ipsi', '0.2', '--prior_contra', '0.1'],
}

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
        time.sleep(30)


def release_gpu(gpu):
    with gpu_lock:
        gpus_in_use.discard(gpu)


def run_one(model, seed):
    gpu = acquire_gpu()
    try:
        outdir = f'{BASE}/results/{model}'
        os.makedirs(outdir, exist_ok=True)
        logfile = f'{outdir}/seed{seed}.txt'
        cmd = ['python', f'{BASE}/train_binary.py',
               '--data_path', '/home/raid-5/MYQ/chest_xray',
               '--backbone', 'resnet50', '--image_size', '512',
               '--batch_size', '16', '--epochs', '30',
               '--lr', '1e-5', '--weight_decay', '1e-4',
               '--num_workers', '4', '--val_split', '0.2',
               '--tta',
               '--seed', str(seed), '--split_file', SPLIT,
               '--no_balance', '--class_weights', '--warmstart', WARMSTART,
               ] + MODELS[model]
        cmd += ['--save_dir', f'{BASE}/results/{model}/ckpt_seed{seed}/']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        with open(logfile, 'w') as f:
            subprocess.run(cmd, cwd=BASE, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(f'[done] {model} seed{seed} on GPU {gpu}', flush=True)
    finally:
        release_gpu(gpu)


def main():
    tasks = [(m, s) for m in MODELS for s in EXTRA_SEEDS if not finished(m, s)]
    print(f'{len(tasks)} extra-seed tasks: {tasks}', flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda t: run_one(*t), tasks))
    print('=== EXTRA SEEDS DONE ===', flush=True)


if __name__ == '__main__':
    main()
