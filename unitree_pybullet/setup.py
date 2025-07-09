# setup.py
from setuptools import setup, find_packages

setup(
    name="unitree_pybullet",
    version="0.1",
    packages=find_packages(),
    include_package_data = True, # データファイルを含めるフラグ
    package_data ={
        "unitree_pybullet":[
            "mocap.txt",
            "data/*",            # plane.urdfなど平置きファイル
            "data/**/*",         # a1/uradf/a1.uradfなどのサブディレクトリ
            ],

    },
)
