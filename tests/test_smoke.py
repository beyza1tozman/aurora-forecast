import aurora
from aurora import config


def test_package_imports():
    assert aurora.__version__


def test_paths_are_inside_repo():
    assert config.RAW_DIR.is_relative_to(config.ROOT)
    assert config.PROCESSED_DIR.is_relative_to(config.ROOT)
