"""Print a quick report of the Python/GPU environment.

Run this first on a new machine: if anything here fails, the rest of the
project will not run either.

Usage (from the project root):
    python src/check_env.py
"""

import importlib
import platform


def check_python() -> None:
    print(f"Python version      : {platform.python_version()}")


def check_torch() -> None:
    """Print torch version, CUDA status, GPU name and free VRAM."""
    try:
        import torch
    except ImportError as error:
        print(f"torch               : NOT INSTALLED ({error})")
        return

    print(f"torch version       : {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available      : {cuda_ok}")
    if not cuda_ok:
        # Without CUDA we would silently train on the CPU, which is far too slow.
        print("GPU name            : (none - torch cannot see a CUDA GPU)")
        return

    print(f"GPU name            : {torch.cuda.get_device_name(0)}")
    # mem_get_info asks the driver directly, so it also counts memory used by
    # other programs (browser, etc.), not just by this Python process.
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    gib = 1024**3
    print(f"VRAM free / total   : {free_bytes / gib:.2f} GiB / {total_bytes / gib:.2f} GiB")


def check_import(module_name: str, label: str) -> None:
    """Try to import a module and print OK + version, or the error."""
    try:
        module = importlib.import_module(module_name)
    except Exception as error:  # broad on purpose: any failure means "not usable"
        print(f"{label:<20}: FAILED ({error})")
        return
    version = getattr(module, "__version__", "unknown version")
    print(f"{label:<20}: OK ({version})")


def main() -> None:
    check_python()
    check_torch()
    check_import("mediapipe", "mediapipe")
    check_import("cv2", "cv2 (OpenCV)")
    check_import("transformers", "transformers")


if __name__ == "__main__":
    main()
