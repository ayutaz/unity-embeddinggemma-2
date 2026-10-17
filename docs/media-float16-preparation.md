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

## 実モデルの準備結果

producer `752a2acd2e517e17501b506df540eeab9d8dac1f`で553モデルを処理し、対応する定数1,110個を半精度化しました。元・派生とも820オブジェクトのサイズとSHA256を独立に照合し、元の全ファイル、モデル以外のファイル、6モジュールの定義が保持されていることを確認しました。保存ファイル合計は8,583,084,520 bytesから5,873,729,560 bytesになりました。

派生descriptor SHA256は`1000f91d47c7def916ace1c7affb5fd01884aab9f3cf064ea62cc48e4d057292`、準備report SHA256は`68d631e308e0faa8aec0a74e5bc8a38353a0a2c225de04add1123dcc9dea4828`です。この生成済みbundleを再利用し、検証コードを更新しただけでは変換を繰り返しません。

CPU専用プロセスは1346.59秒でexit0、ホスト生涯ピークは9,460,596,736 bytesでした。これはEditor起動・準備を含む観測で、推論時間や最低RAM、GPUメモリではありません。詳細は[実モデル準備監査](results/float16-real-preparation-20261018.json)を参照してください。GPU実行、参照精度、順位、連続実行、性能は別途受け入れます。

## 実GPU検証の条件

検証fixtureの`EG_MEDIA_PRECISION`は未指定または`fp32`で元descriptorとcosine >=0.999を維持します。`float16-weights`を明示した場合だけ、上記の派生descriptor・準備reportとcosine >=0.99を使います。形式が不明な場合やハッシュが違う場合は失敗します。有限768次元、norm squaredの許容差0.002、検索の全順位一致・score許容差0.001、実GPUと正常終了の条件は維持します。

ホスト生涯ピークのguardは既定12GiBです。`EG_MEDIA_HOST_LIMIT_GIB`で12〜16GiBの範囲を明示できます。16GiBで測定する場合は、起動前に空き物理メモリが20GiB以上あることを確認・記録します。これは実行時の資源割り当てで、最低RAMの保証や数値基準の変更ではありません。既存12GiB条件での連続実行の中断は、そのまま失敗記録として保持します。

固定ハッシュ・既定FP32の基準・不明形式の拒否・guardの上限を確認する新規17件は、未実装Redから17 Green、skip0、exit0でした。これも実モデルのGPU合格とは別の契約検証です。

## 適用範囲と途中のGPU検証結果

[モジュール別の変換範囲](results/float16-module-coverage-20261018.json)では、テキスト1モデル、画像各181モデル中1モデル、音声各270モデル中15モデルでShort定数が増えました。各モジュールには共有オブジェクトがあるため、件数・容量を合算しません。外部のFP32係数は変わらず、Float16化したbundleであっても全モデル・全演算が半精度になったわけではありません。

SDK `e28a5fde4a1fab246de9cf52ab244b5c6e23d1c4`、固定した派生bundle、Windows Editor / Direct3D12 / GPUComputeで、次の単独プロセスが終了コード0、各1テストpass、fail0、skip0でした。出力768次元の有限性・norm・固定Python参照とのcosineを別途照合しました。

| モジュール | 参照ケース数 | 監査 |
| --- | ---: | --- |
| テキスト | 4 | [実GPU結果](results/text-float16-weights-e28a5fd-20261018.json) |
| 正方形画像 | 2 | [実GPU結果](results/image-square-float16-weights-e28a5fd-20261018.json) |
| 横長画像 | 2 | [実GPU結果](results/image-landscape-float16-weights-e28a5fd-20261018.json) |
| 縦長画像 | 2 | [実GPU結果](results/image-portrait-float16-weights-e28a5fd-20261018.json) |

音声2モジュール、同一プロセスでの連続利用・順位、同一入力の繰り返し性能、新規Git consumerでの実操作は未完了です。stage観測のwall時間はGPU待機・Editorのフレーム待ちを含み、process CPU時間は他のEditor threadも含みます。GPU実行時間は未取得であり、0秒や高速化として報告しません。
