import os

def pytest_addoption(parser):
    os.chdir('tests/test_example')
    parser.addoption("--dataset_config", action="append", default=['flyamer_test_config.json'], help="pytest needs us to explicitly add this so we can load args from a config file")