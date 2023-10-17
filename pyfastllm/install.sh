rm -rf build/ && rm -rf dist/
python3 setup.py sdist bdist_wheel
pip install dist/*.whl --force-reinstall
# python3 examples/test_ops.py # coredump when run with cuda backend