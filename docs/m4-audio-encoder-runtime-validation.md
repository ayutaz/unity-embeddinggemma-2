# 音声encoder Runtime接続のTDD

> 文書整合確認: 2026-10-14。[現在の状態](status.md) / [GPUを主経路とする再開計画](multimodal-tdd-plan.md)。本文の数値・source・実行中／次の作業は記載当時の記録。現在headの実行結果として扱わない。

2026-10-13。`AudioEncoderProfile` / `AudioFeatureEncoder` / `AudioEncodingResult`を追加した。前処理Tensorを借用し、encoderの投影済みsoft tokenを独立したdevice copyとして返す。モデル取得・監査・export、最終768次元APIは別契約。[使い方](../Packages/com.ayutaz.embeddinggemma/Documentation~/audio-feature-encoder.md)、[実測receipt](results/m4-audio-encoder-runtime-contract-windows-20261013.json)。

まずRuntime型が存在しない固定Git `8a7d2f0…`のconsumerへテストのみを配置し、23 assertion failed / passed 0 / skip 0 / CLI exit 1を確認した。未実装型へのreflectionを使い、compile errorを意図したredに数えない。全failureは`AudioEncoderProfile`未実装に由来する。

最小実装を固定Git `235d9fa…`へ保存し、同consumerのrequested / resolved SHA一致、Runtime overlayなしでcompile error / warning 0、23 passed / failed 0 / skip 0 / CLI exit 0を確認した。package assembly全体も165 passed / failed 0 / skip 0。テストを公開APIの直接呼び出しへ整理し、GPU fallback監査は実行した同じModelで行う形へ変更した。固定Git `9de71f7…`へ更新後、compile error / warning 0と23 greenを再確認した。165件の回帰sourceは以前の`235d9fa…`として保持し、新sourceへ付け替えない。

検証は小さいSentis graph。有効frame境界、2段のstride-2によるfeature数、入力 / output shape・型、profile不一致、null / Dispose、worker再利用・出力所有権、実CPU / GPUを確認した。2 framesと3 framesが同じsoft token数でも違うprefixとして拒否する。GPUではCPU計算 / input fallback・readback flushを検査し、借用Tensorをそのままscheduleしてdevice copyを返す。結果の診断readbackはencoder処理と解放後に行い、Runtimeの中間転送とは分ける。

既存Editor PID 16260を再利用し、強制再compile・追加Editor・モデル取得 / 変換は行っていない。開発検証はUnity 6000.3.21f1 / Sentis 2.6.1 / RTX 4070 Ti SUPER / D3D12。公式Test Framework 1.6.0 embeddedの既存補修条件を維持する。今回は既存consumerのGit更新で、最終candidateの空consumer導入ではない。

固定Git cacheのRuntimeファイルとtypedテストもソースへ照合した。最初のバイトhash比較はWindows checkoutのCRLF差で不一致となったが、CRLF → LFだけを正規化した後は両ファイルとも完全一致。Assets内に新Runtime型の定義はなく、package Runtimeの上書きは行っていない。canonical hashをreceiptへ記録する。

実音声encoderの元6信号の数値不合格と新旧参照の不一致はPR #60へ保持している。このAPIの小さいgreenで実モデル・最終音声embedding・sample / Player・必須実機を合格にしない。次は共通backbone / 公式promptとの連結と最終APIのTDD、独立した最終音声参照、実モデル数値gateと全chainの受け入れが必要。
