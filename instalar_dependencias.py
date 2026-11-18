"""Prepare and install the replication environment from the original requirements.

DataPrep 0.4.5 declares a regex upper bound incompatible with tiktoken 0.4.0.
This script builds a local DataPrep wheel with an explicit local version and
updated regex metadata. Its Python source files are preserved byte for byte.
"""

import argparse
import base64
import csv
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parent
DEPENDENCIES = ROOT / "tmp" / "dependencias"
ORIGINAL_NAME = "dataprep-0.4.5-py3-none-any.whl"
LOCAL_VERSION = "0.4.5+autoprep1"
LOCAL_NAME = f"dataprep-{LOCAL_VERSION}-py3-none-any.whl"


def prepare_dataprep():
    DEPENDENCIES.mkdir(parents=True, exist_ok=True)
    original_path = DEPENDENCIES / ORIGINAL_NAME
    if not original_path.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "download", "--no-deps",
             "--dest", str(DEPENDENCIES), "dataprep==0.4.5"],
            check=True,
        )

    expected_sha256 = "646845fc602a95adec0311d89bcf70d260c6c269da3522af9c707887418e8f9c"
    if hashlib.sha256(original_path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("El archivo descargado no coincide con DataPrep 0.4.5 de PyPI.")

    old_info = "dataprep-0.4.5.dist-info/"
    new_info = f"dataprep-{LOCAL_VERSION}.dist-info/"
    entries = {}
    with zipfile.ZipFile(original_path) as original:
        for name in original.namelist():
            if name == old_info + "RECORD":
                continue
            content = original.read(name)
            if name == old_info + "METADATA":
                metadata = content.decode("utf-8")
                old_requirement = "Requires-Dist: regex (>=2021.8.3,<2022.0.0)"
                if metadata.count(old_requirement) != 1:
                    raise ValueError("Los metadatos de DataPrep cambiaron. Revisar antes de continuar.")
                metadata = metadata.replace("Version: 0.4.5\n", f"Version: {LOCAL_VERSION}\n", 1)
                metadata = metadata.replace(
                    old_requirement,
                    "Requires-Dist: regex (>=2022.1.18,<2023.0.0)",
                    1,
                )
                content = metadata.encode("utf-8")
            entries[name.replace(old_info, new_info, 1)] = content

    records = io.StringIO(newline="")
    writer = csv.writer(records, lineterminator="\n")
    for name, content in entries.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
        writer.writerow([name, "sha256=" + digest.decode("ascii"), len(content)])
    writer.writerow([new_info + "RECORD", "", ""])
    entries[new_info + "RECORD"] = records.getvalue().encode("utf-8")

    local_path = DEPENDENCIES / LOCAL_NAME
    with zipfile.ZipFile(local_path, "w", compression=zipfile.ZIP_DEFLATED) as local:
        for name, content in entries.items():
            local.writestr(name, content)

    # Verify that repackaging preserved all DataPrep implementation files.
    with zipfile.ZipFile(original_path) as original, zipfile.ZipFile(local_path) as local:
        source_names = [name for name in original.namelist() if name.startswith("dataprep/")]
        assert source_names, "No se encontraron archivos fuente de DataPrep."
        for name in source_names:
            assert original.read(name) == local.read(name), name
    print("DataPrep preparado. Archivos de implementacion originales verificados.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparar", action="store_true", help="Preparar sin instalar")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 9):
        raise RuntimeError("Ejecuta este archivo con Python 3.9 del entorno autoprep.")

    build_temp = DEPENDENCIES / "build-temp"
    build_temp.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(build_temp)
    os.environ["TEMP"] = str(build_temp)
    os.environ["PIP_CACHE_DIR"] = str(DEPENDENCIES / "pip-cache")

    prepare_dataprep()
    original_bytes = (ROOT / "requirements.txt").read_bytes()
    encoding = "utf-16" if original_bytes.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    original = "\n".join(original_bytes.decode(encoding).splitlines()) + "\n"
    if original.splitlines().count("dataprep==0.4.5") != 1:
        raise ValueError("requirements.txt cambio. Revisar antes de continuar.")
    requirements = original.replace("dataprep==0.4.5", "./tmp/dependencias/" + LOCAL_NAME)
    (ROOT / "requirements-replica.txt").write_text(requirements, encoding="utf-8")
    if args.preparar:
        return

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements-replica.txt",
         "-c", "constraints-replica.txt"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)

    import nltk

    nltk.download(
        "punkt",
        download_dir=str(Path(sys.prefix) / "share" / "nltk_data"),
        quiet=True,
        raise_on_error=True,
    )
    print("Recurso punkt de NLTK instalado.", flush=True)


if __name__ == "__main__":
    main()
