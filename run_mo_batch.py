import argparse, subprocess, pathlib, sys

CONFIGS_ROOT = pathlib.Path(__file__).resolve().parent / "configs"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-dir", required=True,
                   help="Directory to search for configs, relative to configs/")
    p.add_argument("--pattern", default="*.yaml")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--keep-going", action="store_true",
                   help="Continue running remaining configs even if one fails")
    p.add_argument("extra", nargs=argparse.REMAINDER)
    args = p.parse_args()

    cfg_dir = (CONFIGS_ROOT / args.config_dir).resolve()
    if not cfg_dir.is_dir():
        raise SystemExit(f"Config directory not found: {cfg_dir}")

    files = sorted(cfg_dir.rglob(args.pattern))
    if not files:
        raise SystemExit(f"No configs matching {args.pattern} in {cfg_dir}")

    failures = []
    for i, f in enumerate(files, 1):
        # run_mo.py flattens a nested name itself (noisyvis.experiments.config.cli), so unlike
        # run_batch.py this driver passes the nested name straight through.
        config_name = f.with_suffix("").relative_to(CONFIGS_ROOT).as_posix()
        cmd = [
            args.python, "run_mo.py",
            f"--config-name={config_name}",
            *args.extra,
        ]
        print(f"\n>>> [{i}/{len(files)}] {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            failures.append(config_name)
            print(f"!!! {config_name} failed: {e}")
            if not args.keep_going:
                raise

    if failures:
        print(f"\nCompleted with {len(failures)} failure(s): {', '.join(failures)}")
    else:
        print(f"\nAll {len(files)} configs completed successfully.")

if __name__ == "__main__":
    main()
