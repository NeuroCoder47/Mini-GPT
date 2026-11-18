from pathlib import Path


def get_base_dir():
    current_dir = Path(__file__).resolve().parent
    paths_file = current_dir / "paths.txt"

    text = paths_file.read_text(encoding="utf-8")
    lines = text.splitlines()

    data_dir = Path(lines[1]) / "data_train"
    save_dir = Path(lines[2])

    return str(data_dir), str(save_dir)