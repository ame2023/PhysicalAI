# visualize_results.py
"""
学習結果を可視化
episodeに対する累積報酬の推移：r
episodeに対するステップ数：l
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
# monitor.csvの読み込み
repo_root   = Path(__file__).resolve().parents[1]
outputs_dir = repo_root / "outputs"
date = r"2025-07-16\00-48-29" # 可視化したいmonitor.csvのある日付を指定
file_name = 'monitor.csv'
results_path = os.path.join(outputs_dir, date, file_name)
df = pd.read_csv(results_path, names=['r','l','t'], skiprows=2)

# 報酬のプロット
x = range(len(df['r']))
y = df['r'].astype(float)
plt.plot(x, y)
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()

# エピソード長のプロット
x = range(len(df['l']))
y = df['l'].astype(float)
plt.plot(x, y)
plt.xlabel('episode')
plt.ylabel('episode len')
plt.show()