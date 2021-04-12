from setuptools import setup

setup(name='ufl_dnn',
      packages=['ufl_dnn'],
      package_dir={'ufl_dnn': 'ufl_dnn'},
      install_requires=['fenics', 'dolfin-adjoint']
      )
