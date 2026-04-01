 
import pytest
def pytest_collection_modifyitems(items):
    order = {"test_meco_probe": 0, "test_metrics": 1, "test_vector_store": 2}
    items.sort(key=lambda x: order.get(x.fspath.purebasename, 99))