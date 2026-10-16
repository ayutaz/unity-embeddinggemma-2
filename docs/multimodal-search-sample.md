# 画像・音声検索サンプルの作業状況と受け入れ

更新: 2026-10-17。現在の依頼「3まで進める」の手順1・2はPR #31の統合とWindows Playerの固定source検証で完了。手順3の実サンプルは作業中です。元計画の画像・音声API数値検証の完了と、今回のサンプル完成を区別します。

## 実装済み

`Samples~/MultimodalSearch` はPackage ManagerからImportする独立サンプルです。実PNG / JPEG Texture、mono 16 kHz WAV（PCM16 / IEEE float32）とraw float32 PCMを読み込み、GPUで得た768次元vectorを登録します。画像・音声のファイル名とパスは表示用で、テキスト文書としてモデルに与えません。検索文は別のFP32 GPU text moduleで埋め込みます。

準備済みbundleの各モデル・係数・tokenizerをサイズとSHA256で照合します。選択moduleのcontent-addressed objectだけをキャッシュし、warm時に照合済みobjectを再利用します。同じcacheは排他leaseで保護します。切替時は前moduleのGPU処理終了と解放を待ち、登録済みvectorを保持します。ロード失敗・キャンセル後に再実行できます。

`GpuMediaSession` の所有権・キャンセルは9 Red→9 Green、cacheは7 Red→関連34 Green。入力読み込みと基本サンプル操作は10 Red→10 Green。いずれも実Unity6000.3.21f1でskip0。package / validation metadata関連のPython30件も合格しました。これらの単体テストだけでは実モデルの数値合格を証明しません。

## 実モデルの状態

実行Runtime source `02955e3`、bundle producer `a59fb4b` のsquare2画像は、準備済みモデルをRuntime loaderで読み込み、実PNG→GPU前処理→分割encoder→backbone→最終768次元を実行しました。元Document参照とのcosineはそれぞれ0.999999999999714 / 0.999999999999568。NUnit1 pass / fail0 / skip0、session解放とprocess exit0を確認しました。この先行実行のmanaged peak memory値は0で使用できないため、メモリ上限の合格証拠には使いません。後続runnerはWindows native memory counterで監視します。

現行サンプルRuntimeは `5d1a66f`。同じsourceで `text` と `image-square` を再準備済みです。準備はCPU上のgraph保存・ファイル照合であり、モデル推論やGPU数値検証の合格には数えません。画像・音声のモデル演算、元の参照値、学習済み係数、既存の許容差は変更していません。

## 残る受け入れ

1. 実サンプルで画像・音声を登録し、固定した元query / document vectorの順位・scoreと照合する。サンプル操作と直接APIの結果を区別して記録する。
2. 独立したcold / warmプロセスで、転送数・hash・CPU計算fallback拒否・module切替・キャンセル後復帰・lease解放・正常終了を確認する。
3. 3画像profileと2音声profileを準備し、ロード後の実GPU結果を確認する。準備だけの成功を実行成功へ置き換えない。
4. 実行sourceとsample sourceを照合し、実際の画面とsetup文書を確認する。必須CI成功後にfeature PRを統合し、要件ごとの最終監査を行う。

performance campaign、他OS / mobile qualification、tag / releaseは今回の手順1〜3の後です。

## モデル準備と利用

既存の固定参照・qualified graphの入力cacheを持つ検証環境では、opt-in validation packageの `EmbeddingGemma.Validation.Tests.GpuMediaBundlePreparation.Run` をUnity batchmode / nographicsで実行します。環境変数は `EG_ACCEPTANCE_SOURCE`（Runtime / validation両方の固定Git SHA）、`EG_MEDIA_BUNDLE_OUTPUT`（専用出力directory）、`EG_MEDIA_BUNDLE_MODULE`（`text`、`image-square`、`image-landscape`、`image-portrait`、`audio-1119`、`audio-839`）です。

同じ出力には同じproducer sourceだけを使用し、moduleごとの保存を繰り返します。`bundle.json` と `objects/` を一緒に保持し、sampleの準備先へ指定してください。generatorは既存のqualified graph入力を必要とし、空のcheckoutから重みをdownloadするコマンドではありません。入力cacheを持たない環境向けの配布・自動準備手順は未完了です。

利用時にvalidation packageや検証用ONNX assetは不要です。[サンプルの操作と対応入力](../Packages/com.ayutaz.embeddinggemma/Samples~/MultimodalSearch/README.md)を参照してください。
