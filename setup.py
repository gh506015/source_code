from cx_Freeze import setup, Executable

setup(
    name='crawlerSET',
    version='0.1',
    description='test',
    executables=[Executable('crawlerSET.py')]
)