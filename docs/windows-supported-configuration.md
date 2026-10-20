# 初回Windows版の対応構成とメモリ基準

初回正式版のサポート基準は、実SentisのWindows Editor / Playerを確認した下表の構成です。
RAMは64 GB搭載クラス、GPUは16 GB VRAMクラスを保守的なサポート下限として選びます。
これは実測に基づく初回の対応範囲の決定で、より小さい構成が動かないと証明した物理的最小値ではありません。
他GPU・低容量構成の成功を推測して対応表へ追加せず、後続の実測で範囲を広げます。
正式Releaseと新tag consumerの完了判定は別に残っています。

| 項目 | 初回の検証構成 |
| --- | --- |
| OS | Windows 11 x64。実測機はEducation、10.0.22631 / build 22631 |
| Unity | 6000.3.21f1。Windows Playerは既存のWin64 IL2CPP / Release / Highのsource別検証を適用 |
| Sentis | 2.6.1、GPUCompute、Direct3D12 |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER、16 GB VRAMクラス。driver報告の容量は16,376 MiB |
| Driver | Unity実行logは32.0.16.1088、今回のnvidia-smi readbackは610.88 |
| CPU | AMD Ryzen 9 5900X、12 core / 24 logical processor、実CPU比較はAVX2 / 4 thread |
| RAM | 64 GB搭載クラス。OSのphysical totalは68,628,013,056 bytes |

より大きいVRAMや新しいdriverを含め、別のGPU / driverの数値受け入れはこの1構成から保証しません。
macOS / iOS / AndroidはWindows先行公開後の別工程で、未合格です。
比較用の16 lane集計はAVX512命令を要求せず、実AVX2 hostで確認しました。
Python / C++ compiler・比較用CPU helperは開発検証用で、Unity SDKへnative inference pluginを追加していません。

## 選択の根拠

- [Windows機能検証](results/windows-first-stages-1-to-3-20261020.json)は実Editorと、同じSDK実行コードの既存Playerをsource別に確認しています。新tagを新規導入した証拠とは分けます。
- [画像・音声GPU10条件](windows-media-memory-results.md)は40実行と全180区間のallocation sample、正常終了を確認しています。process全体の観測で、保証されたpeak・SDK専有量ではありません。12 query errorと利用不可sampleも保持しています。
- [画像CPU12回](image-CPU-fixed-conditions.md)と[音声CPU8回](native-audio-CPU-results.md)は、元の全数値gate、同じ入力との比較、load / release、正常終了を確認しています。条件別の設定・別frameworkを明示し、これらのCPU process RAMをSDKの最低RAMには換算しません。

成功したEditor検証で使った16 GiB resident / 32 GiB owned-job commitの監視枠は、物理RAMの最低要件ではありません。
以前の16 GiB **commit上限**によるD3D12 device errorも、16 GB搭載PCでの失敗を示しません。
GPUの14 GiB空き容量という起動基準も、機材の最低VRAMを測った結果ではありません。
サンプリングから未観測peakを補間したり、別時刻のdedicated / shared最大値を足して必要容量としたりしません。

初回はこの実測構成を対応範囲として固定し、追加の端末購入や外注を前提にしません。
低容量構成・他GPUの比較や、真の物理下限の探索は未確認事項として保持します。
