# M1 保存・量子化・測定の検証記録

更新日: 2026-10-09。対象: Windows Editor / Unity 6000.3.16f1 / Sentis 2.6.1。
作業ブランチ: `feat/m1-persistence-benchmark`、[PR #4](https://github.com/ayutaz/unity-embeddinggemma-2/pull/4)。
実測コード: `329da925b01f7c9cb0b4d604055f0b9dc214c5c4`。PR #1〜#4は統合済み。
PR #4は2026-10-09 00:32:44 JSTにSquashマージされ、基準mainは `8146107aa77904050c6235866c0cf79ecc80034c`。

## 現在の判定

**Windows EditorのM1受け入れ条件はすべて合格。** `completion-final` は1 passed / failed=skipped=inconclusive=0。
token ID / mask全15件一致、fp32 / Float16重み保存・再読み込み、CPU / GPUCompute各15件（計60比較）、180定常測定値と段階別メモリを確認。
fp32保存前後の最大絶対差は両backendで0。C#単体契約30件も合格。初回NUnit timeoutは失敗履歴として保持し、合格に含めない。
[数値・ソースhash・テスト結果](results/m1-complete-windows-20261009.json)に監査結果を記録した。

## 実測結果

Ryzen 9 5900X / RTX 4070 Ti SUPER / Direct3D12 / Windows 11。batch 1 / length 128。

| 保存形式 | backend | 最小cosine | 初回ms | 中央値ms | p95 ms |
| --- | --- | --- | --- | --- | --- |
| fp32 | CPU | 0.9999999999992348 | 466.29 | 481.43 | 577.86 |
| fp32 | GPUCompute | 0.9999999999997073 | 1298.44 | 50.26 | 68.88 |
| Float16重み | CPU | 0.9999999999991496 | 820.73 | 865.12 | 942.79 |
| Float16重み | GPUCompute | 0.9999999999997065 | 762.07 | 58.20 | 61.71 |

- fp32保存: `1095598044` bytes。Float16重み保存: `553775500` bytes、約49.45%削減。
- 同期再import: 128.72秒。fp32保存: 6.28秒。Float16重み量子化: 1.83秒、保存: 2.69秒。
- 今回の定常推論ではFloat16重み版の方が遅い。保存サイズ削減を速度向上やGPUメモリ削減と読み替えない。
- 保存・ロード・tokenizer準備・Worker作成の個別時間、各ケースcosine / norm / 差分、全サンプル、ファイルSHA-256は数値レポートを参照。

| 測定終了時のEditor全体 | Unity native allocated MiB | Managed heap MiB | Graphics driver MiB |
| --- | --- | --- | --- |
| fp32 / CPU | 2293.4 | 6647.9 | 2307.3 |
| fp32 / GPUCompute | 2287.5 | 7738.1 | 3945.8 |
| Float16重み / CPU | 3353.0 | 5216.1 | 3945.8 |
| Float16重み / GPUCompute | 2310.2 | 5185.7 | 6090.2 |

実行順は上の表の順で、Editor cacheや以前のbackendの割り当てを含む。これらは重なる観測値で、合計・モデル専有量・真のpeak・形式間の独立比較には使わない。
Processのworking set / private bytesはMono APIが0を返し、全サンプルで `processCountersAvailable=false`。未取得であり0バイト使用の意味ではない。

## CIと完了監査

- 実測コード `329da92` の [Python CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37799183105) は4環境各64件合格。
- 同headの [実モデルCI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37799183061) は15件、最小cosine `0.9999997615814209` で合格。
- 配置元の参照 / tokenizer / `.pt2` と保存後2ファイルのSHA-256、37ファイルのLF正規化ソースhash、固定case ID・順序・各基準・中央値 / p95を再計算して確認。
- 以前の実モデルAPI照合のC#ソースhashも一致し、query / document / rawと再利用の証拠を継続利用。生成コード・uv.lock・固定入力は元の基準mainから変更なし。
- 最終PR head `04a970fb07c3f2f1078c3cfb8487c138e3bc9bfe` の [Python CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37801226518) は4環境各64件、[実モデルCI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37801226486) は15件・最小cosine `0.9999998807907104`で合格。
- 統合後main `8146107` の [Python CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37801650704) も4環境各64件合格。実モデルworkflowはmain pushでは起動しない。
- 最終PR headと統合mainのtreeは一致し、実測レポートの37ファイルのLF正規化ソースhashも一致。文書更新で実測commitを書き換えず、Unity測定を再実行しない。

## TDDと失敗履歴

| 対象 | Red / 失敗 | Green / 修正 |
| --- | --- | --- |
| 保存API | `persistence-red` に未実装 `TextModelFile` の9 compile errors | 非破壊fp16保存、保存前後一致、一時ファイルと原子的置換、欠落・破損拒否の5件合格 |
| 保存fixture | 欠落した親ディレクトリでDirectoryNotFoundException、破損モデルの想定ログ未登録 | 親を作成してファイル欠落を検証。Sentisの破損ログをLogAssertで明示。初回失敗XMLを保持 |
| 保存APIのUnity互換性 | 3引数 `File.Move` がUnityの.NETでcompile error | 既存先は `File.Replace`、新規先は `File.Move` に変更 |
| 実モデルランナー | `completion-red` に未実装 `M1CompletionRunner` のcompile error | 保存前後・量子化・両backend・精度・測定を実装 |
| ハーネス | completion scope未実装で4 failed / 18 passed | 対象fixtureと正確な1件・failed / skipped / inconclusive=0を要求。22 passed |
| 実モデル初回 | `completion-first` は全数値照合・測定終了後、NUnit既定180000msを超えて1 failed / 0 skipped | `Timeout(1200000)`を明示。精度閾値・全15件・両backend・測定回数は維持して再実行 |
| メモリ指標 | MonoのProcessカウンタが0を返す。`memory-counter-red` はCaptureMemory未実装でcompile error | process / graphicsの利用可否を明示し、managed heapを別記。既存APIを含めたC#単体契約30件合格 |

`artifacts/unity-harness/` にred / greenのCLI結果、失敗時XML、元ログを保持。
初回測定値は `artifacts/m1-completion-first/results.json`、再実行の構造化結果は `artifacts/unity-harness/completion-final/`。
モデルと大きい実行ログはcommitしない。

## 測定と解釈

- 固定モデルrevision `914f7f89142e33e77833254d9c9b90c3cef7303b`、batch 1 / sequence 128 / 768次元、固定15ケース。
- fp32保存前後はCPU / GPUCompute各15件で最大絶対差 <= 1e-5。Python参照cosine >= 0.999。
- Float16重み保存・ロードは各backend全15件でcosine >= 0.99、有限・768次元・L2 norm 1±0.001。
- 各条件で初回を別測定、固定15件を2周warmup、3周測定（45サンプル）。中央値とnearest-rank p95を記録。
- 同期公開APIのprompting / tokenization / schedule / 最終ベクトルreadbackを含むwall time。純粋なGPU kernel時間ではない。
- importは既存Editorとcache環境での強制同期 `.pt2` 再import。初回インストール時のcold importとは区別。
- Unity native allocated / reserved、managed heap、graphics driverの段階別サンプル。Editor全体の値でありモデル専有や真のpeakではない。相互に重なる指標を合計しない。
- Processカウンタが0なら未取得。graphicsCounterAvailable / processCountersAvailableで判定する。
- Float16は[Sentisの重み量子化](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/quantize-a-model.html)であり、すべての演算がfp16になる保証はない。速度やGPUメモリの改善も実測から判断する。

この測定は1台・1セッションのWindows Editor結果。他プラットフォーム・Player・モバイル・UPM公開は [M2計画](m2-plan.md)で検証する。
