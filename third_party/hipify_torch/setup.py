from setuptools import setup
import setuptools.command.install

## extending the install command functionality.
class install(setuptools.command.install.install):
    def run(self):
        print ("INFO: Installing hipify_torch")
        setuptools.command.install.install.run(self)
        print ("OK: Successfully installed hipify_torch")

cmd_class = {
    "install" : install,
    }

setup(
    name='hipify_torch',
    version='1.0',
    cmdclass=cmd_class,
    packages=['hipify_torch',],
    long_description=open('README.md').read(),
    )
