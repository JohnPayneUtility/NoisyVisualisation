import argparse, subprocess, pathlib, sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-dir", required=True)
    p.add_argument("--pattern", default="*.yaml")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("extra", nargs=argparse.REMAINDER)
    args = p.parse_args()

    cfg_dir = pathlib.Path(args.config_dir).resolve()
    files = sorted(cfg_dir.rglob(args.pattern))
    if not files:
        raise SystemExit(f"No configs matching {args.pattern} in {cfg_dir}")

    for f in files:
        cmd = [
            args.python, "run.py",
            f"--config-path={f.parent.as_posix()}",
            f"--config-name={f.stem}",
            # optional: forward extra hydra/your overrides
            *args.extra,
        ]
        print("\n>>>", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()

