"""프로젝트 공통 경로."""
import os

# .../Disease_Simulation/src/utils/paths.py → 프로젝트 루트
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULT_DIR = os.path.join(PROJECT_ROOT, "result")
LEGACY_DIR = os.path.join(PROJECT_ROOT, "legacy")


def result_path(*parts):
    """result/ 하위 경로를 만들고(필요시 폴더 생성) 반환."""
    p = os.path.join(RESULT_DIR, *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
