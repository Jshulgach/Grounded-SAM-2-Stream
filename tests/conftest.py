import pytest

from grounded_stream.runtime import SearchManager
from tests.fakes import capture, detector
from tests.test_runtime import config


@pytest.fixture
def managers():
    opened = []

    def make(*ids, model="test", **kwargs):
        manager = SearchManager(
            config(*ids, model=model), capture_target=capture, detector_target=detector, **kwargs
        )
        manager.start()
        opened.append(manager)
        return manager

    yield make
    for manager in opened:
        manager.close()
        assert not manager.thread.is_alive()
        assert manager.detector is None
