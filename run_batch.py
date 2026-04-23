import argparse, subprocess, pathlib, sys

CONFIGS_ROOT = pathlib.Path(__file__).resolve().parent / "configs"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-dir", required=True,
                   help="Directory to search for configs, relative to configs/")
    p.add_argument("--pattern", default="*.yaml")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("extra", nargs=argparse.REMAINDER)
    args = p.parse_args()

    cfg_dir = (CONFIGS_ROOT / args.config_dir).resolve()
    if not cfg_dir.is_dir():
        raise SystemExit(f"Config directory not found: {cfg_dir}")

    files = sorted(cfg_dir.rglob(args.pattern))
    if not files:
        raise SystemExit(f"No configs matching {args.pattern} in {cfg_dir}")

    for f in files:
        rel_name = f.with_suffix("").relative_to(CONFIGS_ROOT).as_posix()
        symlink = None
        if "/" in rel_name:
            flat_name = rel_name.replace("/", "__")
            symlink = CONFIGS_ROOT / f"{flat_name}.yaml"
            symlink.symlink_to(f.resolve())
            config_name = flat_name
        else:
            config_name = rel_name
        try:
            cmd = [
                args.python, "run.py",
                f"--config-name={config_name}",
                *args.extra,
            ]
            print("\n>>>", " ".join(cmd))
            subprocess.run(cmd, check=True)
        finally:
            if symlink and symlink.exists():
                symlink.unlink()

if __name__ == "__main__":
    main()
