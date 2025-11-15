import os
import argparse
import subprocess
import shlex

# Silence Ray tip about accelerator env var override when num_gpus=0
# (set before importing Ray so the warning is suppressed).
os.environ.setdefault('RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO', '0')
# Silence Docker CPU detection warning and force multiprocessing CPU count behavior
os.environ.setdefault('RAY_DISABLE_DOCKER_CPU_WARNING', '1')
os.environ.setdefault('RAY_USE_MULTIPROCESSING_CPU_COUNT', '1')

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import sys
import inspect

# Record the caller working directory (driver cwd) so subprocesses launched by Tune
# can run from the original repository directory where `main.py` and package files exist.
DRIVER_CWD = os.getcwd()


def _build_cmd_for_trial(main_py, nproc_per_node, config_path, trial_dir, config):
    # build the torchrun command invoking main.py with CLI args
    cmd = [
        'torchrun',
        f'--nproc_per_node={nproc_per_node}',
        main_py,
    ]
    if config_path:
        # main.py uses snake_case argument names (e.g. --config_path)
        cmd += ['--config_path', config_path]
    # ensure the trial output dir is used (snake_case)
    cmd += ['--output_dir', trial_dir]
    # add hyperparameters from Tune as CLI args (use snake_case names to match main.py)
    for k, v in config.items():
        # keep snake_case keys as-is (e.g. det_token_num -> --det_token_num)
        arg_name = f'--{k}'
        if isinstance(v, bool):
            if v:
                cmd.append(arg_name)
        else:
            cmd += [arg_name, str(v)]
    return cmd


def _get_trial_id():
    # Try Tune API first (may be missing in some Ray versions), then env vars, else fallback to uuid
    try:
        if hasattr(tune, 'get_trial_id'):
            return tune.get_trial_id()
    except Exception:
        pass
    for env_key in ('TUNE_TRIAL_ID', 'RAY_TUNE_TRIAL_ID', 'TUNE_TRIAL_NAME', 'RAY_TUNE_TRIAL_NAME'):
        v = os.environ.get(env_key)
        if v:
            return v
    # fallback
    try:
        import uuid
        return uuid.uuid4().hex
    except Exception:
        return 'unknown'


def make_subprocess_trainable(gpus_per_trial):
    """Return a trainable that launches `torchrun --nproc_per_node=gpus_per_trial main.py` in a subprocess.

    Ray Tune should be instructed to allocate `gpus_per_trial` GPUs per trial via
    `resources_per_trial={"gpu": gpus_per_trial, ...}` when calling `tune.run`.
    """

    def trainable(config, config_path=None, local_dir=None):
        # create trial output dir
        trial_id = _get_trial_id()
        if local_dir is None:
            local_dir = os.path.join(os.getcwd(), 'ray_results')
        trial_dir = os.path.join(local_dir, f'trial_{trial_id}')
        os.makedirs(trial_dir, exist_ok=True)

        # resolve main.py path to the original driver working directory so
        # that when Ray creates temporary working dirs the script path still points
        # to the repository's `main.py` file.
        main_py = os.path.join(DRIVER_CWD, 'main.py')

        # ensure integer number of GPUs per trial
        nproc = int(gpus_per_trial)
        # sanitize certain hyperparams
        # ensure lr > 0 to avoid downstream errors
        if 'lr' in config:
            try:
                lr_val = float(config['lr'])
                if lr_val <= 0 or (not (lr_val > 0)):
                    config['lr'] = 1e-6
            except Exception:
                config['lr'] = 1e-6

        # write sampled config for debugging
        try:
            import json
            with open(os.path.join(trial_dir, 'sampled_config.json'), 'w') as cf:
                json.dump(config, cf, indent=2)
        except Exception:
            pass

        # build command using repr for floats to preserve precision
        cmd = _build_cmd_for_trial(main_py, nproc, config_path, trial_dir, {k: (repr(v) if isinstance(v, float) else v) for k, v in config.items()})

        print('Launching trial subprocess:', ' '.join(shlex.quote(c) for c in cmd))

        # open a logfile for the child process stdout/stderr so we can inspect failures
        proc_log_path = os.path.join(trial_dir, 'proc.log')
        proc_log = open(proc_log_path, 'wb')

        # If single-process (nproc == 1), run Python directly (no torchrun). This
        # avoids torchrun staging issues and is faster for single-GPU trials.
        # Prepare subprocess environment: ensure CUDA_VISIBLE_DEVICES from Ray is preserved
        proc_env = os.environ.copy()
        # Ray injects CUDA_VISIBLE_DEVICES into the worker process; propagate it to child
        cuda_dev = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cuda_dev is not None:
            proc_env['CUDA_VISIBLE_DEVICES'] = cuda_dev

        # Log assigned devices for debugging
        print(f"[trial {trial_id}] CUDA_VISIBLE_DEVICES={proc_env.get('CUDA_VISIBLE_DEVICES')}")

        if nproc <= 1:
            launch_cmd = [sys.executable, main_py]
            # append args from cmd after the script (cmd contains torchrun options before)
            # find position of main_py in cmd and append subsequent args, but here cmd has main_py at index 2
            launch_cmd += cmd[3:]
            proc = subprocess.Popen(launch_cmd, stdout=proc_log, stderr=subprocess.STDOUT, env=proc_env, cwd=DRIVER_CWD)
        else:
            # start subprocess (non-blocking) using torchrun; run from driver cwd so main.py is available
            proc = subprocess.Popen(cmd, stdout=proc_log, stderr=subprocess.STDOUT, env=proc_env, cwd=DRIVER_CWD)

        # monitor the trial's log file for metrics and report to Tune
        import time
        import json
        last_reported_epoch = -1
        log_path = os.path.join(trial_dir, 'log.txt')

        mean_ap_reported = False
        try:
            while True:
                # if process finished, break after one final parse
                if proc.poll() is not None:
                    # final read
                    if os.path.exists(log_path):
                        with open(log_path, 'r') as f:
                            lines = f.read().strip().splitlines()
                        if lines:
                            try:
                                last = json.loads(lines[-1])
                                # extract mean AP if present
                                if 'test_coco_eval_bbox' in last:
                                    bbox = last['test_coco_eval_bbox']
                                    if isinstance(bbox, (list, tuple)) and len(bbox) > 0:
                                        tune.report({"mean_ap": bbox[0], "done": True})
                                        mean_ap_reported = True
                            except Exception:
                                pass
                    # wait for subprocess to fully exit
                    proc.wait()
                    break

                # if log exists, parse last line
                if os.path.exists(log_path):
                    try:
                        with open(log_path, 'r') as f:
                            lines = f.read().strip().splitlines()
                        if lines:
                            last = json.loads(lines[-1])
                            # infer epoch and mean AP if available
                            epoch = last.get('epoch', None)
                            if 'test_coco_eval_bbox' in last:
                                bbox = last['test_coco_eval_bbox']
                                if isinstance(bbox, (list, tuple)) and len(bbox) > 0:
                                    mean_ap = bbox[0]
                                    # avoid duplicate reports
                                    if epoch is None or epoch > last_reported_epoch:
                                        tune.report({"mean_ap": mean_ap, "epoch": epoch if epoch is not None else -1})
                                        mean_ap_reported = True
                                        if epoch is not None:
                                            last_reported_epoch = epoch
                    except Exception:
                        pass

                time.sleep(20)
        except KeyboardInterrupt:
            proc.terminate()
            raise
        finally:
            # ensure process cleaned up
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
            try:
                proc_log.close()
            except Exception:
                pass

            # If we never reported mean_ap, report a sentinel and include a small proc.log excerpt
            if not mean_ap_reported:
                proc_excerpt = ''
                try:
                    if os.path.exists(proc_log_path):
                        with open(proc_log_path, 'rb') as pf:
                            pf.seek(0, os.SEEK_END)
                            size = pf.tell()
                            tail_size = min(size, 8192)
                            pf.seek(size - tail_size)
                            tail = pf.read().decode('utf-8', errors='replace')
                            proc_excerpt = tail
                except Exception:
                    proc_excerpt = ''
                # report a sentinel mean_ap so Tune sees the metric and can continue
                tune.report({"mean_ap": -1.0, "failed": 1, "proc_log_tail": proc_excerpt[:4096], "done": True})

    return trainable


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Ray Tune search for YOLOS hyperparameters')
    parser.add_argument('--num-samples', type=int, default=20, help='Number of trials to sample')
    parser.add_argument('--max-epochs', type=int, default=12, help='Max epochs (budget) used by ASHA')
    parser.add_argument('--gpus-per-trial', type=int, default=1)
    parser.add_argument('--cpus-per-trial', type=int, default=8)
    parser.add_argument('--local-dir', type=str, default='search')
    parser.add_argument('--config-path', type=str, default='configs/dinov3_small_freeze.yaml', help='optional base yaml config path')
    args = parser.parse_args()

    # Search space for requested hyperparameters
    search_space = {
        # learning rate: log-uniform between 1e-6 and 1e-3
        'lr': tune.loguniform(1e-6, 1e-3),
        # eos_coef: classification no-object weight
        'eos_coef': tune.uniform(0.01, 0.5),
        # set_cost_class: matching cost for classes
        'set_cost_class': tune.uniform(0.1, 5.0),
        # det_token_num: number of detection tokens
        'det_token_num': tune.choice([50, 75, 100, 150, 200]),
        # eval_size: evaluation image size
        'eval_size': tune.choice([512, 640, 800, 1024]),
    }

    scheduler = ASHAScheduler(
        max_t=args.max_epochs,
        grace_period=1,
        reduction_factor=3
    )

    trainable = make_subprocess_trainable(args.gpus_per_trial)

    # prepare an absolute storage/local dir for Ray (pyarrow requires a URI)
    storage_dir_abs = os.path.abspath(args.local_dir)
    os.makedirs(storage_dir_abs, exist_ok=True)

    # Explicitly init Ray to avoid automatic init message and to allow
    # passing configuration if needed later.
    ray.init(ignore_reinit_error=True)

    # Use file:// URI for storage_path so pyarrow can parse it correctly
    storage_uri = f"file://{storage_dir_abs}"

    # Determine available GPUs according to Ray and compute a safe concurrent trial limit.
    try:
        avail = ray.available_resources()
        total_gpus = int(avail.get('GPU', 0))
    except Exception:
        total_gpus = 0

    # Compute max concurrent trials so we don't oversubscribe GPUs.
    if args.gpus_per_trial > 0 and total_gpus > 0:
        max_concurrent = max(1, total_gpus // args.gpus_per_trial)
    else:
        # if Ray doesn't expose GPUs or gpus_per_trial==0, fallback to 1 concurrent trial
        max_concurrent = 1

    print(f"Ray reports total_gpus={total_gpus}; setting max_concurrent_trials={max_concurrent}")

    # Build common kwargs for tune.run
    run_kwargs = dict(
        run_or_experiment=tune.with_parameters(trainable, config_path=args.config_path, local_dir=storage_dir_abs),
        resources_per_trial={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
        metric="mean_ap",
        mode="max",
        num_samples=args.num_samples,
        config=search_space,
        scheduler=scheduler,
        storage_path=storage_uri,
        name="yolos_ray_search",
        raise_on_failed_trial=False,
        max_concurrent_trials=max_concurrent,
    )

    # Some Ray versions don't accept 'queue_trials' or accept different arg names.
    # Inspect signature and call accordingly with a graceful fallback.
    try:
        sig = inspect.signature(tune.run)
        if 'queue_trials' in sig.parameters:
            tune.run(**run_kwargs, queue_trials=True)
        else:
            print('tune.run does not support queue_trials in this Ray version; running without queueing')
            tune.run(**run_kwargs)
    except Exception:
        # Last-resort: try calling and fall back if TypeError about unexpected arg
        try:
            tune.run(**run_kwargs, queue_trials=True)
        except TypeError:
            print('tune.run rejected queue_trials; retrying without it')
            tune.run(**run_kwargs)
