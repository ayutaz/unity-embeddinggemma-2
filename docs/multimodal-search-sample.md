# 画像・音声検索サンプルの作業状況と受け入れ

更新: 2026-10-17。現在の依頼「3まで進める」の手順1・2はPR #31の統合とWindows Playerの固定source検証で完了。手順3は実装・実モデル・画面の検証を終え、全体回帰と最終PR / CI確認を進めています。元計画の画像・音声API数値検証の完了と、今回のサンプル完成を区別します。

## 実装済み

`Samples~/MultimodalSearch` はPackage ManagerからImportする独立サンプルです。実PNG / JPEG Texture、mono 16 kHz WAV（PCM16 / IEEE float32）とraw float32 PCMを読み込み、GPUで得た768次元vectorを登録します。画像・音声のファイル名とパスは表示用で、テキスト文書としてモデルに与えません。検索文は別のFP32 GPU text moduleで埋め込みます。

準備済みbundleの各モデル・係数・tokenizerをサイズとSHA256で照合します。選択moduleのcontent-addressed objectだけをキャッシュし、warm時に照合済みobjectを再利用します。同じcacheは排他leaseで保護します。切替時は前moduleのGPU処理終了と解放を待ち、登録済みvectorを保持します。ロード失敗・キャンセル後に再実行できます。

`GpuMediaSession` の所有権・キャンセルは9 Red→9 Green、cacheは7 Red→関連34 Green。入力読み込みと基本サンプル操作は10 Red→10 Green。いずれも実Unity6000.3.21f1でskip0。package / validation metadata関連のPython30件も合格しました。これらの単体テストだけでは実モデルの数値合格を証明しません。

## 実モデルの状態

実行Runtime source `02955e3`、bundle producer `a59fb4b` のsquare2画像は、準備済みモデルをRuntime loaderで読み込み、実PNG→GPU前処理→分割encoder→backbone→最終768次元を実行しました。元Document参照とのcosineはそれぞれ0.999999999999714 / 0.999999999999568。NUnit1 pass / fail0 / skip0、session解放とprocess exit0を確認しました。この先行実行のmanaged peak memory値は0で使用できないため、メモリ上限の合格証拠には使いません。後続runnerはWindows native memory counterで監視します。

先行サンプルRuntime `5d1a66f` と同じsourceで `text` と `image-square` を再準備しました。準備はCPU上のgraph保存・ファイル照合であり、モデル推論やGPU数値検証の合格には数えません。画像・音声のモデル演算、元の参照値、学習済み係数、既存の許容差は変更していません。

現在はproducer `5d1a66f` の6 moduleすべてを準備済みです。作業環境では `artifacts/step3-media-bundle-5d1a66f` がその準備先です。モデル8,583,084,520 bytes・820 objectはGitへ含めません。

[square2画像の独立cold / warm監査](results/multimodal-search-square-sample-cold-warm-20261017.json)では、Runtime51ファイルとimportしたsample8ファイルを固定Git sourceへ照合しました。両起動で全4 queryの順位が一致し、最大score差は1.0183251353268474e-7。warmはdescriptor1件だけを取得し、298 objectを再利用しました。GPU-stageキャンセル後に同じsampleで復帰し、正常終了しています。

## 音声キャンセル後のメモリ増加と修正

Runtime `ee90c2a` の実sampleで、silence PCMとsquare画像を登録したcold起動は合格しましたが、音声GPU処理のキャンセル後に再実行するとhost peakが16,757,751,808 bytesへ増え、12 GiB guardで停止しました。これは不合格として保持します。最初の検証用PCM filename誤りも別の失敗として保持し、音声の推論成功に数えません。

`AudioPreprocessor.ProcessAsync` は入力・owned outputをGPU fence完了まで保持し、コピー元output・free buffer・compiled operator contextを解放します。`AudioEmbedder.EmbedAsync` はこの非同期前処理を待ってからencoderへ進みます。キャンセル・Disposeでもfence完了を待ち、借用stageを先に解放しません。実GPUの小さい契約4 Red→4 Green、前処理を待つAPIの3 Red→関連31 Greenを確認しました。

修正Runtime `5ca5fdd` の[実sampleによる独立監査](results/multimodal-search-mixed-sample-cache-retirement-20261017.json)は、音声GPU-stageキャンセル→再登録→画像module切替→固定4 query→解放→exit0が合格。host peakは10,190,360,576 bytes（約9.49 GiB）で同じ12 GiB guard内でした。Worker / kernel / tensorの解放後fieldも検査しています。画像・音声の最終vector captureは修正前のcold合格と全bit一致し、元query / documentに対する最大score差は1.7913466188179683e-7です。warmの音声・画像・textはいずれもdescriptor1件のみ取得しました。

これらは固定source・Windows Editor / Unity6000.3.21f1 / Sentis2.6.1 / RTX4070 Ti SUPER / D3D12での数値・所有権検証です。performance benchmarkや全端末の合格ではありません。

[準備済み全profileの独立監査](results/multimodal-prepared-media-all-profiles-gpu-20261017.json)は、画像6ケース・3 profileと音声6ケース・2 profileの全12条件が合格。最小cosineは0.9999998052055729、5プロセスともNUnit1 pass / fail0 / skip0・exit0でした。最大host peakは12,090,900,480 bytesで12 GiB guard内です。対象は元Document最終768次元で、全encoderの要素別比較やQuery / Rawを再実行したという主張ではありません。

[実PlayMode画面の監査](results/multimodal-search-playmode-ui-20261017.json)では、importした `MultimodalSearch.unity` を開き、実音声・画像を登録して固定4 queryを検索しました。EditModeの独立監査とvector / scoreが一致し、Worker・kernel・tensorとcache leaseの解放、exit0を確認。日本語・英語の表示をとで目視しました。入力とボタンの手動操作は別途実施していません。

## CIの独立したCPU対照失敗

PR #83の旧CI38931525134ではIntel Xeon8370Cのnyquistで、native全体とnative分割carryのbit一致が失敗し、ONNX実行前に停止しました。fresh native対元参照も1,840値が元許容差を超過しており、GPU不合格の証拠へ置き換えません。

CPU対照の候補対策として、必須workflowのnative BLAS branchを `MKL_CBWR=COMPATIBLE`、dynamic threadを無効へ固定しました。元のnative bit一致gate、5 partの実carry、元ONNX許容差は維持し、不一致carry自体も保存するようにしました。関連Python59件は合格。PR head `ee90c2a` の[CI38933177757](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38933177757)は全15 jobが成功しました。code path・alignment・threadが丸め順へ影響し得るという[IntelのCNR説明](https://www.intel.com/content/www/us/en/developer/archive/training/conditional-numerical-reproducibility-cnr.html)を参照した対策で、CPU failureの原因確定とは表現しません。後続のGPU前処理修正を含む最終PR headのCIは別gateです。

## 残る受け入れ

1. 最終sourceのpackageと両sampleの全体回帰を実行する。
2. 最終PR headの必須CI成功後にfeature PRを統合し、要件ごとの最終監査を行う。

performance campaign、他OS / mobile qualification、tag / releaseは今回の手順1〜3の後です。

## モデル準備と利用

既存の固定参照・qualified graphの入力cacheを持つ検証環境では、opt-in validation packageの `EmbeddingGemma.Validation.Tests.GpuMediaBundlePreparation.Run` をUnity batchmode / nographicsで実行します。環境変数は `EG_ACCEPTANCE_SOURCE`（Runtime / validation両方の固定Git SHA）、`EG_MEDIA_BUNDLE_OUTPUT`（専用出力directory）、`EG_MEDIA_BUNDLE_MODULE`（`text`、`image-square`、`image-landscape`、`image-portrait`、`audio-1119`、`audio-839`）です。

同じ出力には同じproducer sourceだけを使用し、moduleごとの保存を繰り返します。`bundle.json` と `objects/` を一緒に保持し、sampleの準備先へ指定してください。generatorは既存のqualified graph入力を必要とし、空のcheckoutから重みをdownloadするコマンドではありません。入力cacheを持たない環境向けの配布・自動準備手順は未完了です。

この作業環境では追加の変換は不要です。Package Managerで **Image and Audio Search** をImportし、`MultimodalSearch.unity`を開いてPlayします。準備先にrepository内の `artifacts/step3-media-bundle-5d1a66f` の絶対パスを指定してください。PNG / WAV / .f32の絶対パスを指定して対応profileで登録し、検索文を入力します。通常の利用でvalidation packageを導入する必要はありません。

利用時にvalidation packageや検証用ONNX assetは不要です。[サンプルの操作と対応入力](../Packages/com.ayutaz.embeddinggemma/Samples~/MultimodalSearch/README.md)を参照してください。
