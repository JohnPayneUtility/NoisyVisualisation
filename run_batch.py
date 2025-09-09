import argparse, subprocess, pathlib, sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-dir", required=True, help="Folder containing .yaml configs")
    p.add_argument("--pattern", default="*.yaml", help="Glob to select configs")
    p.add_argument("--python", default=sys.executable, help="Python to use")
    p.add_argument("extra", nargs=argparse.REMAINDER,
                   help="Extra overrides forwarded to Hydra (e.g. seed=1 hydra.verbose=true)")
    args = p.parse_args()

    cfg_dir = pathlib.Path(args.config_dir).resolve()
    files = sorted(cfg_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No configs matching {args.pattern} in {cfg_dir}")

    for f in files:
        name = f.stem
        cmd = [
            args.python, "run.py",
            f"--config-name={name}",
            f"+hydra.searchpath=[file://{cfg_dir.as_posix()}]"
        ] + args.extra
        print("\n>>>", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
