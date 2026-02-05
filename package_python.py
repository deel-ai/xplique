import argparse
import re

import requests
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version


def filename_implies_compat(filename: str, target: Version) -> bool:
    t_major, t_minor = target.release[:2]

    # # Look for py3 or py2.py3 in filename
    # if "py3" in filename and t_major == 3:
    #     return True
    # if "py2.py3" in filename:
    #     return True

    # Look for cpXY like cp38 or cp310 in filename
    for m in re.finditer(r"cp(\d)(\d+)", filename):
        major = int(m.group(1))
        minor = int(m.group(2))
        if (major, minor) == (t_major, t_minor):
            return True
    return False


def get_compatible_versions(package: str, py_version: Version, cponly: bool) -> list[str]:
    url = f"https://pypi.org/pypi/{package}/json"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data: dict = r.json()

    compatible = []
    # For each release of package
    for ver, files in data.get("releases", {}).items():
        ver: str
        files: list[dict]
        # Assert version is valid, skip if not (e.g., "0.1.0a0.dev0+githash")
        try:
            Version(ver)
        except InvalidVersion:
            continue

        # Inside each release, check if any platform-specific file is compatible with
        # the target Python version
        release_ok = False  # Will be True if any file is compatible
        for f in files:
            if not cponly:
                req = f.get("requires_python") or ""
                if req:
                    try:
                        spec = SpecifierSet(req)
                        if spec.contains(py_version):
                            release_ok = True
                            break
                    except Exception:
                        pass
            # Fallback to filename heuristics (wheels)
            if filename_implies_compat(f.get("filename", ""), py_version):
                release_ok = True
                break

        # # Fallback to project-level requires_python as no release-specific info found
        # if not release_ok and project_requires:
        #     try:
        #         spec = SpecifierSet(project_requires)
        #         if spec.contains(py_version):
        #             release_ok = True
        #     except Exception:
        #         pass
        if release_ok:
            compatible.append(ver)

    # Sort using packaging.Version
    compatible = sorted(compatible, key=lambda v: Version(v))
    return compatible


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check compatible package versions for a given Python version."
    )
    parser.add_argument(
        "package",
        nargs="?",
        default="pytorch",
        type=str,
        help="Package name (e.g., tensorflow)",
    )
    parser.add_argument(
        "python_version",
        nargs="?",
        default="3.14",
        type=str,
        help="Python version (e.g., 3.14)",
    )
    parser.add_argument(
        "--cponly",
        default=False,
        type=bool,
        help="Only consider files with cpXY in filename for compatibility",
    )
    args = parser.parse_args()

    pkg: str = args.package
    py = Version(args.python_version)
    cponly: bool = args.cponly
    print(f"Compatible {pkg} versions for Python {py}:")
    versions = get_compatible_versions(pkg, py, cponly)
    if len(versions) >= 2:
        print(f"First {versions[0]} and last {versions[-1]}")
    print(" ".join([v for v in versions]))
