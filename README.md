# 使い方

## TODO
* 報酬関数の設計(重み調整)
* PDゲインの調整
* 可操作度項の重み調整
* 

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
* best_modelのテスト
```bash
python test_trained_model.py --model-path outputs/2025-07-16/00-40-36/best_model.zip --results-dir ./my_test_results/2025-07-16/00-40-36 --episodes 20
```

等速再生
```bash
python test_trained_model.py \
  -m outputs/2025-07-16/00-40-36/ppo_a1.zip \
  --realtime --speed 1.0
```
倍速再生
```bash
python test_trained_model.py -m outputs/.../ppo_a1.zip --realtime --speed 2.0
```

学習過程の可視化(src/)
* 可視化したいCSVの指定
```bash
python visualize_results.py --monitor-csv outputs/2025-07-22/00-07-25/monitor.csv
```
* PNGとして保存(例：my_test_results/monitor.csvと同階層に保存)
```bash
python visualize_results.py --monitor-csv my_test_results/monitor.csv --save-png
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
annotated-types        0.7.0
antlr4-python3-runtime 4.9.3
certifi                2025.7.14
charset-normalizer     3.4.2
click                  8.2.1
cloudpickle            3.1.1
colorama               0.4.6
contourpy              1.3.3
cycler                 0.12.1
Farama-Notifications   0.0.4
filelock               3.18.0
fonttools              4.59.0
fsspec                 2025.7.0
gitdb                  4.0.12
GitPython              3.1.45
gymnasium              1.2.0
hydra-core             1.3.2
idna                   3.10
imageio                2.37.0
imageio-ffmpeg         0.6.0
Jinja2                 3.1.6
kiwisolver             1.4.8
markdown-it-py         3.0.0
MarkupSafe             3.0.2
matplotlib             3.10.3
mdurl                  0.1.2
mpmath                 1.3.0
networkx               3.5
numpy                  2.3.2
omegaconf              2.3.0
packaging              25.0
pandas                 2.3.1
pillow                 11.3.0
pip                    24.0
platformdirs           4.3.8
protobuf               6.31.1
pybullet               3.2.7
pydantic               2.11.7
pydantic_core          2.33.2
Pygments               2.19.2
pyparsing              3.2.3
python-dateutil        2.9.0.post0
pytz                   2025.2
PyYAML                 6.0.2
requests               2.32.4
rich                   14.1.0
sentry-sdk             2.33.2
setuptools             80.9.0
six                    1.17.0
smmap                  5.0.2
stable_baselines3      2.7.0
sympy                  1.14.0
torch                  2.7.1+cu128
torchaudio             2.7.1+cu128
torchvision            0.22.1+cu128
tqdm                   4.67.1
typing_extensions      4.14.1
typing-inspection      0.4.1
tzdata                 2025.2
urllib3                2.5.0
wandb                  0.21.0
wheel                  0.45.1
```

```bash
pip install -r requirements.txt
```
