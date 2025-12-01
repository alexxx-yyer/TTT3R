import sys
import os


def add_path_to_dust3r(ckpt):
    here_path = os.path.dirname(os.path.abspath(ckpt))
    repo_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(repo_root, "src")

    # Ensure both the checkpoint directory and local source tree are importable.
    for p in (here_path, repo_root, src_path):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
