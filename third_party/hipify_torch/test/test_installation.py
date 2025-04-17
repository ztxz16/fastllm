import unittest

class TestHipifyInstallation(unittest.TestCase):
    def test_hipify_torch_installation(self):
        try:
            from hipify_torch import hipify_python
        except ImportError:
            print ("ERROR: please install hipify_torch using setup.py install")
            raise ImportError('Install hipify_torch module')

if __name__ == '__main__':
    unittest.main() 
