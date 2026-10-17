# Float16重みの派生bundle

`GpuMediaFloat16Preparation`は、固定FP32 bundleからSentis 2.6.1の`ModelQuantizer.QuantizeWeights(Float16)`を使って、対応するモデル定数を変換します。外部の係数・tokenizerなど、モデル以外のファイルは同じビット列のままです。全演算をhalfへ変更する処理ではありません。

元bundleのdescriptor SHA256は`8363f9a63a95b307fbbb835f3551a41f6fa5a1d21aca3017670e6f678647e955`、元モデルrevisionは`914f7f89142e33e77833254d9c9b90c3cef7303b`です。全入力のサイズとハッシュを照合してから出力を作成します。入力と重なる出力先、所有記録のない出力先、異なるproducerからの再開は拒否します。共有ファイルは一度だけ処理し、モデルの外部入出力名・型・shapeを保持します。

検証用consumerへこのvalidation packageを導入し、作業ツリーが固定producerの内容と一致することを確認したうえで実行します。既存のFP32モデルやLibraryをGitへ追加しません。

```powershell
$env:EG_ACCEPTANCE_SOURCE = git rev-parse HEAD
$env:EG_FLOAT16_SOURCE = '<検証済みFP32 bundleの絶対パス>'
$env:EG_FLOAT16_OUTPUT = '<入力と重ならない新しい出力先>'
& '<Unity 6000.3.21f1のUnity.exe>' -batchmode -nographics `
  -projectPath '<validation packageを導入したconsumer>' `
  -executeMethod EmbeddingGemma.Validation.Tests.GpuMediaFloat16Preparation.PrepareFromEnvironment `
  -quit -logFile '<新しいログファイル>'
```

実行は時間・ホストメモリを制限した専用プロセスで監視します。重い変換は入力とUnity環境を利用できるGitHub Actionsを優先します。現在の検証済みbundleはローカルにあるため、公開取得ルートが整うまで、同じ固定入力を使うローカル変換を実施します。

`float16-preparation.json`に、入力descriptor・producer、各入力モデルと派生モデルのSHA256、Short定数数、処理したモデル・module数、最終descriptor SHA256を記録します。中断時は元のproducer・入力を維持して再実行します。既存オブジェクトもハッシュを再検証し、異なるmodule内容で公開済みdescriptorを上書きしません。

新規9件は未実装RedからGreenとなりました。既存Writer契約も含め、正式validation assemblyは15件、legacy Writer assemblyは6件、プロセス全体は21件pass、fail0、skip0、exit0です。これは小さなモデルでの準備処理の検証です。実モデルの生成、GPU実行、数値・順位・速度・連続実行の合格とは区別します。
