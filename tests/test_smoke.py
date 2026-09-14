from forestwatch.config import settings
from forestwatch.dataset_scanner import scan_dataset, inventory_summary

def test_dataset_scans():
    inventory = scan_dataset(settings.dataset_path)
    assert len(inventory) > 0
    assert inventory_summary(inventory)["counts"].get("ndvi", 0) > 0
