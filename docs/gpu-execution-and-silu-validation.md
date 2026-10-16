# GPU負荷計測とSiLU修正後の実モデル照合

2026-10-14。「ここからの計画」の1（GPU確認・負荷計測）と2（音声SiLU修正の実モデル照合）までを実施した。1は指定範囲で合格、2は全件照合を終えて数値不合格。次の数値修正と全chainには着手していない。[現在の状態](status.md) / [全体計画](multimodal-tdd-plan.md)。変更はDraft PR #81、main未統合。

## 1: GPU確認と負荷

固定Git consumerのrequested / resolvedはRuntime・validationとも `b796e504367fb9165a4992ff036a205fb2e7d429`。Unity 6000.3.16f1 / Sentis 2.6.1、RTX 4070 Ti SUPER / Direct3D12。GPU専用変更を含むC# compile成功、tinyと実square-0の2テストはpass2 / fail0 / skip0 / exit0。requested / actual backendともGPUCompute、計算または入力CPU fallback 0、readback flush 0。診断の最終出力だけCPUへ読み戻した。数値encoder gateを合格とするテストではない。

実画像は元の保存モデル・参照JSON・入力hashを監査し、以前重かった補償候補と同じprojection chunk64 / norm chunk4へ変換した。層数は1823→25265。[固定source・全段階のreceipt](results/m2-gpu-phase-profile-windows-20261014.json)。

| 段階 | 経過秒 | CPU秒 | 終了時プロセスworking set (GiB) |
| --- | ---: | ---: | ---: |
| 読み込み・hash確認・入力準備 | 3.935 | 3.516 | 3.00 |
| 補償graph変換 | 13.130 | 11.609 | 3.06 |
| Worker作成・fallback確認 | 0.067 | 0.078 | 3.06 |
| 入力GPU upload | 0.008 | 0.000 | 3.07 |
| Schedule・出力backend確認 | 2.074 | 0.469 | 3.93 |
| 最終readback・有限性確認 | 1.301 | 0.000 | 3.94 |
| Dispose | 0.010 | 0.016 | 3.94 |

値はreceiptの小数を丸めた。load + transformは全7段階約20.53秒の約83%で、CPU時間は約15.13秒。今回の候補ではGPU推論以外のモデル準備が大きい。Scheduleは投入を含むホスト側時間、最終readbackにはGPU完了待機が含まれるので、各値をGPU kernel時間とは呼ばない。

プロセスpeak working setは約3.94 GiB。これはEditor・モデル・CPU/GPU準備を含むWindowsプロセスの値で、モデルだけのRAMやGPU VRAMではない。単発・square 1件であり、定常性能・旧graphとの比較・全profile・利用率時系列を確認した結果ではない。Unity全processの起動から終了まで約58.44秒。Accelerator接続timeoutと終了時Persistent allocation警告も保持し、原因・長期resource安定性は未解決としている。

TDDは計測機能なしの1 red → 実装後、MonoのProcess.WorkingSet64が0のため2 red → Windows native memory counterに修正して2 green。0を有効な計測として扱わない。外部300秒上限は `tools/embeddinggemma_tools/bounded_process.py` で実装し、小さい3件でtimeout停止・自然終了0・非zero失敗・既存証拠上書き拒否を確認した。実Unityは上限前に正常終了。古いignored launcherの無制限待機は変更していない。

## 2: 音声SiLU修正の実モデル照合

[Actions 38578451611](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38578451611)、source `a8033bbfdcfb6bcac9936a5cd9a3e119352e121c`、artifact `11901382904`。修正済み標準ONNX SiLU `Neg→Exp→Add→Div` の全encoderを生成・独立評価し、元の6信号・2 profileを照合した。後段stage / backbone生成は今回の手動実行で省いた。workflow設定は1 red→関連11 green。

6信号は各所属profileで1回ずつ、合計6件。6信号×2 profileの12件ではない。5信号はframes-1119-valid-1119、padded_noiseはframes-1119-valid-839。

| 信号 | 不一致要素数 | 最大許容差比 |
| --- | ---: | ---: |
| silence | 588 | 9.764 |
| impulse | 1410 | 22.794 |
| sine | 715 | 13.384 |
| nyquist | 1529 | 11.467 |
| noise | 294 | 5.074 |
| padded_noise | 248 | 5.074 |

全6件不合格、Actionsもfailure。小さいSiLUの除算順序修正は全encoderの数値差を解消しなかった。[失敗receipt](results/m4-division-silu-real-onnx-failed-20261014.json)。

元producer `4edbe17a8105233c5aaa1514faa2862b08a4f7e0` と比較し、モデル入力ファイル・library版・6信号の入力・mask hashは一致。公式native特徴量も全6件bit一致した。旧goldenと `atol=1e-5` / `rtol=1e-4` を維持。ONNX hashは意図した演算変更により両profileとも変わっているため、旧graph一致を含むpinned-contextの総合flagはfalse。これを参照値の数値不一致と混同しない。

4.94 GBのartifact全体をローカル取得せず、必要な4 JSONだけrange取得（約25.20 MB）して確認した。大きい参照JSON・モデル・重み・生ログはGitへ含めず、source / hash / 集計を小さいreceiptへ保存した。これはPython ONNXの検証であり、Sentis音声GPUや最終768次元APIの合格ではない。

次は計画3の一つのGPU入力・最初の数値差を原因別TDDへ落とす。今回の指定範囲では実行しない。
