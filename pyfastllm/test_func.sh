pip uninstall -y fastllm
rm -rf fastllm/pyfastllm.cpython-310-x86_64-linux-gnu.so
rm -rf build/
python3 build_libs.py
python3 setup.py sdist bdist_wheel
pip install dist/fastllm-0.1.4-py3-none-any.whl
python3 demo/test_ops.py