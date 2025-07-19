# 使い方

## 学習

以下のコマンドをターミナルで実行して、強化学習を開始します。

```bash
python main.py
```

## 学習済みモデルのテスト

以下のコマンド例のように、学習済みモデルをテストできます。

```bash
python test_trained_model.py \
  --model-path outputs/2025-07-16/00-40-36/ppo_a1.zip \
  --results-dir ./my_test_results/2025-07-16/00-40-36 \
  --episodes 20
```

# ファイル説明

* `main.py`：強化学習を実行するスクリプト
* `test_trained_model.py`：学習済みモデルをロードしてテストするスクリプト
* src/ のファイル
* `visualize_results.py`:outputsに保存されているmonitor.csvを可視化

### 環境定義

`unitree_pybullet/unitree_pybullet/` 以下に環境クラスとユーティリティ関数があります。

* `QuadGymEnv.py`：シミュレーション環境クラスの定義
* `manip_utils.py`：可操作度（Yoshikawa 指標など）の計算ユーティリティ

# 仮想環境の作成およびアクティベート

以下のコマンドで Python 仮想環境を作成し、有効化します。

```bash
# 仮想環境の作成
target_dir=venv
python -m venv $target_dir

# Windows の場合
source $target_dir/Scripts/activate
# macOS/Linux の場合
source $target_dir/bin/activate
```

# 環境構築

依存パッケージをインストールします。バージョンは参考例です。

```text
Package                Version
---------------------- -----------
accelerate             0.29.3
annotated-types        0.7.0
antlr4-python3-runtime 4.9.3
asttokens              2.4.1
certifi                2024.2.2
charset-normalizer     3.3.2
click                  8.1.8
cloudpickle            3.0.0
colorama               0.4.6
comm                   0.2.2
contourpy              1.2.1
cycler                 0.12.1
debugpy                1.8.1
decorator              5.1.1
diffusers              0.27.2
docker-pycreds         0.4.0
executing              2.0.1
Farama-Notifications   0.0.4
filelock               3.13.4
fonttools              4.53.0
fsspec                 2024.3.1
gitdb                  4.0.12
GitPython              3.1.44
gym                    0.9.7
gym-notices            0.0.8
gymnasium              1.1.1
huggingface-hub        0.22.2
hydra-core             1.3.2
idna                   3.7
importlib_metadata     7.1.0
ipykernel              6.29.4
ipython                8.23.0
jedi                   0.19.1
Jinja2                 3.1.3
joblib                 1.4.2
JSAnimation            0.1
jupyter_client         8.6.1
jupyter_core           5.7.2
kiwisolver             1.4.5
MarkupSafe             2.1.5
matplotlib             3.10.3
matplotlib-inline      0.1.7
mpmath                 1.3.0
nest-asyncio           1.6.0
networkx               3.3
numpy                  1.26.4
omegaconf              2.3.0
packaging              24.0
pandas                 2.2.2
parso                  0.8.4
pillow                 10.3.0
pip                    24.0
platformdirs           4.2.0
prompt-toolkit         3.0.43
protobuf               6.30.2
psutil                 5.9.8
pure-eval              0.2.2
pybullet               3.2.7
pydantic               2.11.4
pydantic_core          2.33.2
pyglet                 1.2.4
Pygments               2.17.2
pyparsing              3.1.2
python-dateutil        2.9.0.post0
pytz                   2024.1
pywin32                306
PyYAML                 6.0.1
pyzmq                  26.0.0
regex                  2024.4.16
requests               2.31.0
safetensors            0.4.3
scikit-learn           1.4.2
scipy                  1.13.0
sentry-sdk             2.27.0
setproctitle           1.3.6
setuptools             80.3.1
six                    1.16.0
smmap                  5.0.2
stable_baselines3      2.6.0
stack-data             0.6.3
sympy                  1.14.0
threadpoolctl          3.5.0
tokenizers             0.19.1
torch                  2.7.1
torchaudio             2.2.2
torchvision            0.17.2
tornado                6.4
tqdm                   4.66.2
traitlets              5.14.2
transformers           4.40.0
typing_extensions      4.13.2
typing-inspection      0.4.0
tzdata                 2024.1
urllib3                2.2.1
wandb                  0.19.10
wcwidth                0.2.13
wheel                  0.45.1
zipp                   3.18.1
```

```bash
pip install -r requirements.txt
```
