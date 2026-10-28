"""Reject missing library files or unexpected files in PyPI distributions."""

from pathlib import Path
import tarfile
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    package_files = {
        path.relative_to(root).as_posix()
        for path in (root / "temporal_networks").rglob("*")
        if path.is_file() and (path.suffix in {".py", ".pyi"} or path.name == "py.typed")
    }
    source_files = {
        "README.md", "LICENSE", "pyproject.toml", "MANIFEST.in", "PKG-INFO", "setup.cfg",
    }
    metadata_files = {"METADATA", "WHEEL", "RECORD", "top_level.txt", "licenses/LICENSE"}
    archives = sorted((root / "dist").glob("*"))
    if len(archives) != 2 or {p.suffix for p in archives} != {".gz", ".whl"}:
        raise SystemExit("dist/ must contain exactly one source archive and one wheel")
    for archive in archives:
        if archive.name.endswith(".tar.gz"):
            with tarfile.open(archive) as source:
                files = {
                    member.name.split("/", 1)[1]: source.extractfile(member).read()
                    for member in source.getmembers() if member.isfile()
                }
            allowed = package_files | source_files | {
                "temporal_networks.egg-info/" + name
                for name in ("PKG-INFO", "SOURCES.txt", "dependency_links.txt",
                             "requires.txt", "top_level.txt")
            }
            required = package_files | source_files
        elif archive.suffix == ".whl":
            with zipfile.ZipFile(archive) as wheel:
                files = {name: wheel.read(name) for name in wheel.namelist()}
            prefix = archive.name.split("-py3-", 1)[0] + ".dist-info/"
            allowed = required = package_files | {prefix + name for name in metadata_files}
        else:
            raise SystemExit(f"Unexpected archive: {archive.name}")
        unexpected, missing = set(files) - allowed, required - set(files)
        if unexpected or missing:
            raise SystemExit(f"{archive.name}: unexpected={sorted(unexpected)}, missing={sorted(missing)}")
        for name in package_files:
            if files[name] != (root / name).read_bytes():
                raise SystemExit(f"{archive.name}: stale library file {name}")
        print(f"{archive.name}: verified {len(files)} files")


if __name__ == "__main__":
    main()
