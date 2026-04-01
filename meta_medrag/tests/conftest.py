 
import pytest

def pytest_collection_modifyitems(items):
    # Force test_meco_probe first, test_metrics second, test_vector_store last
    order = {"test_meco_probe": 0, "test_metrics": 1, "test_vector_store": 2}
    items.sort(key=lambda x: order.get(x.fspath.purebasename, 99))