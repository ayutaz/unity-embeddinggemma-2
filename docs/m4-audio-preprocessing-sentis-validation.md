# M4音声前処理のSentis CPU / GPU受け入れ

> 文書整合確認: 2026-10-14。[現在の状態](status.md) / [GPUを主経路とする再開計画](multimodal-tdd-plan.md)。本文の数値・source・実行中／次の作業は記載当時の記録。現在headの実行結果として扱わない。

実行日: 2026-10-13。Unity 6000.3.21f1 / Sentis 2.6.1、Windows、RTX 4070 Ti SUPER / Direct3D12の既存consumerで、`uloop run-tests`によるcompileとEditMode 2件が成功した。failed / skipped / inconclusiveは0。両backendそれぞれ6信号を比較した。[小さい結果記録](results/m4-audio-preprocessing-sentis-windows-20261013.json)。

## 固定入力と判定

- producerはmain `e35470212f616b020c0b150a4c515a007f0b937f`の[Actions run 38320110239](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38320110239)。生成済み前処理graphと全参照ファイルをmanifestのサイズ・SHA-256で監査した。モデルrevisionは`914f7f89142e33e77833254d9c9b90c3cef7303b`。
- 公式`Gemma4AudioFeatureExtractor`の参照を使用。利用者の音声ではなく、silence / impulse / sine / nyquist / noise / padded_noiseの手続き生成PCMを使った。
- 入力は16kHz mono、179200 samples、出力は`[1,1119,128]` featureと`[1,1119]` frame mask。graphをSentisへimportし、`.sentis`として保存 / 再読込した。
- producerで事前に固定した要素単位の`atol=0.004` / `rtol=1e-4`を維持し、全featureが有限・許容差以内、全frame maskが完全一致した。CPU / GPUの最大絶対誤差はともに`0.0038890838623046875`、不一致要素0。
- GPU入力をupload後、実worker / feature / mask backendはGPUCompute。CPU計算・入力fallback indicesとGPU readback flush indicesは0件。最後のfeature / mask取得は参照比較のための診断readbackであり、encoder / backbone間の本番接続ではない。

## 範囲と残る条件

前処理export自体の35 red → 契約green・監査 / 回帰110 passedは[元のTDD記録](m4-audio-preprocessing-validation.md)に保持する。今回追加したのは既存exportに対する独立参照の受け入れfixtureであり、新しい演算実装のredを捏造しない。

fixtureは開発sourceを既存consumerのEditor test assemblyへ配置して実行した。固定Gitからの最終candidate導入、製品PCM Runtime API、音声encoder、最終768次元、実音声の意味検索・マイク・Player / 他OSは未検証。`encoders_validated=false` / `end_to_end_validated=false`を維持する。前処理の成功でPR #40のencoder数値不合格を解消扱いにしない。

Test Runnerは生成assetをcleanupしていない旨のWarningを1件出した。生成assetと大きい入力はignore対象であり、モデル・PCM・大きいartifactはコミットしていない。既存Libraryと単一Editorを再利用した結果で、空cache導入やdomain reload高速化の証拠ではない。

次はA2の入力境界・所有権をTDDで実装し、A4の独立Python実モデルgateを修正してから、A3 → A4 → A5をGPU tensorで接続する。段階単独の診断取得と、本番で中間CPU readbackを発生させない条件を別に検証する。
