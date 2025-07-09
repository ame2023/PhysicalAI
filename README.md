# C++コンパイラのインストール
１．公式サイトから「Build Tools for Visual Studio 2022」をダウンロード
    https://visualstudio.microsoft.com/visual-cpp-build-tools/
    ここで入手できるのは「Visual Studio Installer」の小さなランチャーです。

２．Installer を起動 → ワークロード画面
    ・たいてい「Workloads」（左ペイン）にある
        ・「Desktop development with C++」
        　 これを選ぶと、MSVC コンパイラ（v143〜）や Windows SDK、CMake など一式入ります。
    ・「C++ build tools」だけ少数コンポーネントで入れたい場合は、右側タブの Individual components から
        ・「MSVC v143 – VS 2022 C++ x64/x86 build tools」
        ・「Windows 10 SDK」
        をチェックしてインストールしてください。

pybulletでC++のコンパイルが必要になるため上記をインストールしておく

# 仮想環境の作り方
cmd上で
py -3.10 -m venv venv   #バージョンを3.10系に指定。ただし予めローカルPCにインストールしておく必要がある

バージョン指定しない場合
python -m venv venv 

# 仮想環境のアクティベート
.\venv\Scripts\activate


