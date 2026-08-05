import argparse, subprocess, pathlib, sys

CONFIGS_ROOT = pathlib.Path(__file__).resolve().parent / "configs"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-dir", default="LONs/penalty_colons",
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

    files = sorted(cfg_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No configs matching {args.pattern} in {cfg_dir}")

    failures = []
    for i, f in enumerate(files, 1):
        config_name = f.stem
        cmd = [
            args.python, "run_colon_parallel.py",
            f"--config-path={cfg_dir}",
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
