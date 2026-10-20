# M4音声前処理の固定profile exportと参照

> 設計・検証の履歴です。本文の状態・実行中・次の作業は記載当時のものです。現在の対応範囲は[検証状況](status.md)、残作業は[ロードマップ](roadmap.md)を参照してください。固定sourceの失敗・未実行も証拠として保持します。

更新日: 2026-10-13。[M4計画](m4-audio-embedder-validation.md)のA1 / A3のPython exportを進める。**音声encoder、Unity実GPU、最終embedding、検索 / 再生 / 収録はまだ合格していない**。

## 先に固定した参照と数値条件

固定revision `914f7f89142e33e77833254d9c9b90c3cef7303b`、Transformers 5.19.0のGemma4AudioFeatureExtractorを独立参照にする。16 kHz mono、window 320 / hop 160 / FFT 512 / mel 128、periodic Hann、先頭160 sampleをzero padするsemicausal framing。frameは321 sampleで切り出し、先頭320 sampleをwindow / FFTへ使う。magnitudeをHTK melへ掛け、`log(mel + 0.001)`を計算し、frame末端sampleのmaskで無効frameをzeroにする。

実装前に同じ1024 sampleの5信号で公式NumPyとTorch FP32 FFTを比較した。

| 信号 | 最大絶対差 |
| --- | --- |
| silence | 0 |
| impulse | 0.000000477 |
| 440 Hz sine | 0.002474308 |
| Nyquist alternation | 0.001113415 |
| deterministic noise | 0.000000567 |

このbaselineを記録してから、featureのrtol `1e-4` / atol `0.004`をテストへ固定した。実装失敗に合わせた緩和はしていない。これは前処理featureの条件であり、A5の最終embedding cosine >= .999 / 採用Float16重み >= .99の条件は維持する。

## 実装とTDD

- `tools/embeddinggemma_tools/audio.py`にreal tensorだけのradix-two FFTを実装。固定gather / slice / reshape / arithmeticを使い、complex / FFT / unfold演算を保存graphへ残さない。windowとmel係数は固定版の公式extractorを使う。
- batch 1 / 固定sample数、FP32 PCMの[-1,1]と有限値、int64 sample maskのbinary・実sample prefix・right zero paddingを検査する。最小実長161 sample。対応外processor設定を拒否し、serialized frame / hop / FFT設定を黙って無視しない。
- outputはFP32 featureとbinary int32 frame mask。保存graphを再ロードし、公式NumPy参照の全feature / maskを照合する。wrong output、dtype、mask、stale成功報告を監査する。
- waveform・mask・境界 / 型 / 設定・保存graphで35 red → 35 green。参照 / hash / CLI / workflowで5 red、serialized設定4 red、保存output契約2 redを確認。最終の音声・画像・backbone・workflow関連回帰110 passed、skip 0。
- ローカルは1024 sampleの小さい信号のみ。標準長の参照 / graphはActionsへ寄せ、Pythonはtools/とuvを使う。CPUでのexport数値成功をSentis GPU成功と扱わない。

eagerのhost入力検査はexport graphの外にある。Unity入力APIにも同じ契約を実装し、PCM取得 / uploadと、それ以後のGPU前処理を分ける。

## 必須Actions gateと成果物

```powershell
cd tools
uv run --locked python -m embeddinggemma_tools.audio --output ../artifacts/audio-preprocessing
```

同revisionのprocessor configだけを取得し、ここではencoder重みを取得・ロードしない。default profileは179,200 sample（11.2秒）、1,119 mel frame。silence / impulse / sine / Nyquist / noise / right padded noiseの6ケースを公式featureと保存graphへ渡す。truncationや独自chunk平均は追加しない。

`audio-reference.json`、`audio-validation.json`、source / processor / library版・hash、PCM FP32 / mask int32 / reference feature FP32のlittle-endian binary、`.pt2`を別artifactへ保存する。信号はコードで生成し、利用者の録音を収集・保存しない。`stage=audio_preprocessing_only`、`unity_validated=false`、`encoders_validated=false`を維持する。成功receiptは全ケース合格後だけ保存し、失敗logと部分artifactを残す。

Actions結果は取得後に追記する。次はSentis importと実CPU / GPUの全feature / mask照合、A4の音声encoder / projector、A5の共通backbone・prompt / pooling / 最終vectorを進める。マイク入力の動作と固定PCMの精度、各実機の権限・memory・終了も別に検証する。正式リリースは行わない。
