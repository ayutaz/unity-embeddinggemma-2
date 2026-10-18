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

## 適用範囲と単独GPU検証結果

[モジュール別の変換範囲](results/float16-module-coverage-20261018.json)では、テキスト1モデル、画像各181モデル中1モデル、音声各270モデル中15モデルでShort定数が増えました。各モジュールには共有オブジェクトがあるため、件数・容量を合算しません。外部のFP32係数は変わらず、Float16化したbundleであっても全モデル・全演算が半精度になったわけではありません。

SDK `e28a5fde4a1fab246de9cf52ab244b5c6e23d1c4`、固定した派生bundle、Windows Editor / Direct3D12 / GPUComputeで、次の単独プロセスが終了コード0、各1テストpass、fail0、skip0でした。出力768次元の有限性・norm・固定Python参照とのcosineを別途照合しました。

| モジュール | 参照ケース数 | 監査 |
| --- | ---: | --- |
| テキスト | 4 | [実GPU結果](results/text-float16-weights-e28a5fd-20261018.json) |
| 正方形画像 | 2 | [実GPU結果](results/image-square-float16-weights-e28a5fd-20261018.json) |
| 横長画像 | 2 | [実GPU結果](results/image-landscape-float16-weights-e28a5fd-20261018.json) |
| 縦長画像 | 2 | [実GPU結果](results/image-portrait-float16-weights-e28a5fd-20261018.json) |
| 音声 / 1119 valid frames | 5 | [実GPU結果](results/audio-1119-float16-weights-e28a5fd-20261018.json) |
| 音声 / 839 valid frames | 1 | [実GPU結果](results/audio-839-float16-weights-e28a5fd-20261018.json) |

全6モジュール・固定16参照ケースの集約は[単独GPU検証監査](results/all-six-module-float16-e28a5fd-acceptance-20261018.json)を参照してください。各モジュールは独立Editorプロセスで実行しました。同一プロセスでの連続利用・順位、同一入力の繰り返し性能、新規Git consumerでの実操作は未完了です。stage観測のwall時間はGPU待機・Editorのフレーム待ちを含み、process CPU時間は他のEditor threadも含みます。GPU実行時間は未取得であり、0秒や高速化として報告しません。

## FP32の空cache・EditMode連続検証

同じSDKで16GiBのホスト割り当てを明示し、空のモデルcacheから画像3種類・音声2種類・日英4検索を行うfixtureと、直接テキスト4件のfixtureを連続実行しました。サンプルfixtureは900,000msの上限で失敗し、画像3件・音声1件の登録だけが完了しました。検索は未実行です。続くテキスト4件は合格し、プロセス全体は1 pass / 1 fail / skip0 / exit2でした。参照値との独立照合に通った途中結果を、連続検証全体の合格へ数えません。[失敗と部分結果の監査](results/fp32-e28a5fd-cold-editmode-timeout-20261018.json)を参照してください。

この起動の初期Refreshは81.042秒、実importは0件で、再利用projectの観測です。内包するcompile時間51.352秒・domain reload時間3.217秒を合計へ加算せず、GPU推論時間にも置き換えません。Windowsのプロセス別GPU allocationは244サンプルを記録しました。dedicated/shared allocationと実常駐VRAM、未観測のピーク、最低VRAMは区別します。

fixtureの時間超過後に、継続した処理から破棄済みComputeShaderへのアクセスも記録されました。これだけで製品のmemory leakやGPU演算の失敗原因は確定しません。続く確認には、同じ5媒体・4検索・数値／順位／解放条件を持つ既存PlayMode描画fixtureを使います。このfixtureの実行上限もEditModeと同じ900,000msとし、Unity Test Frameworkの既定180,000msとの違いを明示します。Editorの更新待ちが寄与する程度は未測定であり、PlayModeで実行するだけで高速化を主張しません。

## FP32のPlayMode連続検証

SDK `e28a5fd`、同じ固定FP32 bundle、明示した16GiB割り当てで、PlayModeの既存描画fixtureと直接テキストfixtureが2 pass / fail0 / skip0 / exit0となりました。画像3件・音声2件の登録と固定Python参照との比較、日英4クエリの全順位・score許容差0.001、GPU処理中のcancel/retry、module切替、Worker/kernel/tensorとcache leaseの解放を確認しました。続くテキスト4件も独立に照合しています。[連続実行と描画の監査](results/combined-fp32-e28a5fd-16gib-playmode-20261018.json)を参照してください。

4枚の1280×800 PNGでは、合成画像3種類のプレビューと、日本語の音声再生ボタン・順位・scoreの描画を目視確認しました。スクロール領域の最終行が一部だけ表示されるクエリもあります。実際のマウス／キーボード操作、再生・停止、実マイク、新規Git consumerの受け入れとは別の結果です。

プロセス全体は836.64秒で、Editor起動も含みます。初回module準備中にUnityを前面へ切り替えた観測を記録しており、先のEditMode実行との時間差をモデル高速化へ帰属させません。ホスト生涯ピークは14,559,334,400 bytesでした。これはEditorを含む当該processの観測で、最低RAMやVRAMではありません。単独／連続のFloat16比較、同一入力の反復・各stageの実GPU時間は引き続き未完了です。

## Float16のPlayMode連続検証

同じSDKと固定したFloat16重みbundleでも、画像3件・音声2件の登録、日英4検索の全順位・score許容差0.001、GPU処理中のcancel/retry、切替・解放を確認しました。続く直接テキスト4件と合わせて2 pass / fail0 / skip0 / exit0です。最終ベクトルと固定参照の比較、全順位とscore、4枚のPNGのハッシュを独立に照合しました。[Float16の連続実行・描画監査](results/combined-float16-weights-e28a5fd-16gib-playmode-20261018.json)を参照してください。

4枚の描画画像では、画像プレビュー・日本語の再生ボタン・順位表示を目視確認しました。丸めたscoreが同じに見える行も、内部の全精度scoreと固定順位で判定しています。実際のマウス／キーボード操作や音声の再生・停止はこのfixtureでは実施していません。

プロセス全体の観測は583.01秒、ホスト生涯ピークは11,157,610,496 bytesでした。初回GPU登録中に前面化しており、FP32実行とは前面化のタイミングやcache条件を統制した速度比較ではありません。これらの値から高速化率、最低RAM、最低VRAMは主張しません。単独・連続の固定数値／順位／解放は両精度で確認できましたが、同一入力のfirst/warm/repeatと各stageのCPU／実GPU時間、新規Git consumerでのnative操作、公開cold取得、許可後の実マイクは残っています。

両精度の計8枚に対する目視確認は、再生成される数値監査と分けて[描画の確認記録](results/playmode-render-visual-review-20261018.json)へ保存しています。確認したPNGのSHA256・サイズと、未実施のnative操作・再生の範囲を記録しています。
## Explicit descriptor preparation for a new bundle

`GpuMediaFloat16Preparation.PreparePinnedFromEnvironment` requires
`EG_FLOAT16_DESCRIPTOR_SHA256` (the reviewed source bundle's lowercase SHA256)
and `EG_ACCEPTANCE_SOURCE` (the fixed producer commit). Set `EG_FLOAT16_SOURCE`
and `EG_FLOAT16_OUTPUT` to separate private directories. Unlike the original
entry point, this entry point has no default descriptor pin. Missing,
malformed or mismatched pins fail before output creation. The existing
preparation checks still validate every source object and preserve non-model
coefficient bytes and external model bindings.

The [explicit-pin TDD checkpoint](results/float16-explicit-pin-tdd-20261018.json)
observed ten intended missing-entry-point failures with nine existing passes,
then 19 passes with no skips and process exit 0. This is local-file,
no-graphics metadata/weight-conversion testing on tiny fixtures. It does not
establish real-model GPU accuracy, half-precision computation or speedup.

The subsequent [new-candidate real-model checkpoint](results/audio-candidate-float16-bundle-player-c18b4e5-20261018.json)
at fixed Git source `c18b4e5` prepared all 285 unique audio models. All 434 source
and derived objects, non-model coefficient bytes, two module rewrites, model
description strings and a second private copy were independently checked.
The actual Windows IL2CPP / Release / High stripping Player then passed both
audio profiles with 18 base and 18 in-process warm vectors, both 270-Worker
GPU audits, cancellation recovery, release and process exit 0. Independent
original-reference recalculation rejected 24 corrupted controls. The binary
was built at `89b1ab9` with the same SDK Runtime Git tree; no rebuild is claimed.
This is not independent warm-launch, all-half arithmetic, a speedup measurement,
other-platform acceptance or a formal release.
