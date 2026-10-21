# M4音声encoder / projectorの固定profile export

> 設計・検証の履歴です。本文の状態・実行中・次の作業は記載当時のものです。現在の対応範囲は[検証状況](status.md)、残作業は[ロードマップ](roadmap.md)を参照してください。固定sourceの失敗・未実行も証拠として保持します。

## 2026-10-14の追加状況

2026-10-15の最新GPU証拠は[元通常5ケース](results/m4-scalar-moment-fma-full-profile-gpu-20261015.json)と[元padded_noise / adapter](results/m4-padded-audio-and-gpu-encoder-adapter-20261015.json)。固定sourceで元6ケース / 2 profileの許容差超過0、実GPUCompute・計算/入力CPU fallback0・中間flush0、fail0 / skip0。元golden / 学習済み係数 / 許容差を保持した。最新headでの全件再実行、CPU合格、実PCM→768 API全条件・sample / Player合格へは拡張しない。詳しくは[M4現在計画](m4-audio-embedder-validation.md)。以下は旧候補の履歴。

最新のnorm1 / dynamic16候補も元12数値条件すべて不合格。[追加receipt](results/m4-norm1-dynamic16-full-encoder-failed-20261014.json)。標準ONNX SiLU順序修正の実モデルは未検証。

## 当時の実測・実装記録

更新日: 2026-10-13。当時のA1 / A4（音声encoder）の実装を進める。**この出力は共通本体へ渡す512次元soft tokenであり、最終768次元embeddingではない**。Sentis実CPU / GPU、共通本体・検索 / 再生 / マイク、各実機は未検証。正式リリースは作らない。

## 実装とTDD

- 公式Gemma4AudioModelとEmbeddingGemma2MultimodalEmbedderのaudio prefixだけをstrict FP32 loadする。text / visionモデルは構築・ロードしない。projector欠落は拒否する。
- 固定batch 1、FP32 log-magnitude mel、非空のint64 binary prefix frame maskを契約にする。2段のsubsample convolution、relative position、chunk attention、Conformerのfeed-forward / causal convolution、output projectionと公式multimodal projectorを維持する。
- chunk contextの`Unfold`を固定Gatherへ、encoder後のboolean選択をprefix sliceへ変換する。非finite、shape / dtype不一致、nonzero右padding、非prefix maskを拒否する。host検査は保存graphに含めないため、Unity入力にも同じ監査が必要。
- 未実装encoder / loader・入力契約の20件の意図したredを観測した。SDPA参照の固定と不適切なeager設定の拒否は追加2 red。関連小さい回帰99 passed。
- 参照生成・artifact監査・stale成功除去・保存済み誤出力拒否・CLIの8 red、必須workflow接続の1 redを確認し、参照準備9 passed。最終関連回帰は145 passed / skip 0。
- 初回workflowテストのWindows文字コードエラーはUTF-8指定で修正し、その後にencoder未接続を拒否するassertion失敗を確認した。文字コードエラーをアルゴリズムredには数えない。

### 参照のattention設定

固定transformers 5.19.0の公式audioモデルの既定値は`sdpa`である。初期fixtureでvisionと同じ`eager`を強制すると、mask作成がfloatの加算maskを返し、audio attentionのboolean mask処理と一致しなかった。小さい入力で公式maskのdtypeと内容、Unfold対Gatherの差0を確認して原因を分離した。

audioの参照は公式既定のSDPA boolean mask契約に固定する。期待値は公式audioモデルの出力から独立に生成し、独自adapterから生成しない。eager強制時の5件の数値失敗は参照設定の調査結果として残し、許容差を緩めた合格とはしない。loaderをSDPAに固定し、異なる設定のwrapperは拒否する。export自体は公式chunk attentionの数式を通常tensor演算で保存する。

## 必須Actions gate

```powershell
cd tools
uv run --locked python -m embeddinggemma_tools prepare-audio-encoder --preprocessing ../artifacts/audio-preprocessing --output ../artifacts/audio-encoder
```

先行する[音声前処理gate](m4-audio-preprocessing-validation.md)の**公式NumPy参照特徴量**を入力とする。独自前処理graphの出力をencoderの期待入力にしない。6種類の生成PCM由来の入力をframe maskごとにprofile化し、同じshapeでもvalid prefix長が異なる場合は別graphへ分ける。

- 前処理の完了stage / source / model revision、全ファイルsize / SHA-256、参照JSON・feature / maskのmanifest掲載、snapshotとprocessor hashの一致、finite・prefix・paddingをweights load前に監査する。
- 保存`.pt2`を再ロードし、公式audio encoder / projectorの全featureとのshape / FP32 / finite、cosine >= 0.999999、elementwise rtol `1e-4` / atol `1e-5`を要求する。これらは画像exportと同じPython export照合条件であり、最終embeddingの条件を置き換えない。
- 元モデル / processor / 前処理参照 / graph / binary / 出力参照のhash、lockしたlibrary版、profile・source SHAを記録する。成功は全ケース照合後にのみ保存し、途中失敗時はstale成功を除去する。
- 別artifactを3日保持し、重み・graph・大きいtensorをGit / LFSへ含めない。`stage=audio_encoder_projector_only`、`unity_validated=false`、`end_to_end_validated=false`を維持する。

最初の実encoder [CI run 38317139304](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38317139304)（head `d4f7b2668d3108349edca12ed8a55b7d07d4028e`）は、無音の保存graph対公式encoder照合で失敗した。minimum cosine `0.9999998211860657`はcosine条件を満たすが、elementwise条件を超えた。成功receiptは生成していない。Python4環境・lint・packageは成功したが、実encoder gateとRequired CIは失敗である。

この失敗を小さいテストで追跡できるよう、失敗時の独立したeager / 保存graph / 公式出力の最大差・許容差比・不一致要素数・graph hash / 演算子・source / caseを記録する診断と、stale失敗receipt除去を追加した。2件の意図したred → 関連9 passed / skip 0。元の数値許容差は維持し、再実行で原因を分離する。診断binaryはActions artifactだけに保存する。Python export成功をSentis / GPU / 最終vectorの成功には数えない。

### 保存グラフの固定attention係数

[診断CI 38318322669](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38318322669)では、無音のeager wrapper対公式出力の最大差は0、保存graph対eagerの最大差は `0.00020956993103027344`、固定許容差比は `8.21683406829834`、不一致209要素だった。保存graphの演算分解に起因する差として調査を進める。

小さいFP32演算診断で、native `Softplus`とCore ATenの `log1p(exp(x))`に最大差 `9.5367431640625e-7`を確認した（入力範囲 -10〜10 の10001点）。音声attentionの学習済みper-dimension query係数は入力に依存しないため、公式native Softplusで一度計算してbufferとして保存する。`query * q_scale * coefficient`の乗算順序は維持する。これは学習更新を扱わない固定重みの推論exportである。

保存attentionにSoftplus分解が残り、native係数を保持できないことを1件の意図したassertion redで確認した。修正後は保存係数の完全一致、演算子監査、変更したhidden入力に対する独立公式attention照合を含め、encoder / 参照監査の関連31 passed / skip 0（44.88秒）。native SiLUにも小さい分解差を観測しており、この係数修正だけで実encoderの不合格が解消したとはまだ判断しない。元の実モデル許容差を維持したActions再照合が必要。

[係数修正後CI 38319899052](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38319899052)（head `1eabc0f0d0d2c2b2bd936b820fc9c2baf3a27fd9`）も無音の要素単位条件で不合格。eager対公式差0、保存対eager最大差 `0.00015294551849365234`、許容差比 `6.410955429077148`、不一致430要素。最大差は減ったが合格ではなく、不一致数も増えている。Python4環境・lint・packageは合格、実encoder / Required CIは失敗。

### 音声SiLUの丸め順序

固定[PyTorch 2.14.1のnative CPU SiLU](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cpu/Activation.cpp#L1098-L1128)は `x / (1 + exp(-x))`で計算する。Core ATen既定の `x * sigmoid(x)`は、同じ数学的関数でも逆数を丸めてから乗算するためFP32値が異なる。10001点の小さい診断で、既定分解は最大差 `9.5367431640625e-7` / 不一致2204点、divisionの順序では最大差 `4.76837158203125e-7` / 不一致426点だった。これらは小さい演算結果であり、実encoderの受け入れではない。

固定反例 `8.01` / `5.0`とsigned値で、保存audio graphのnative演算完全一致テストを追加し、2 / 7要素が異なる意図したredを確認した。音声exportだけにnative division順序のSiLU分解を適用し、その他のCore ATen分解は維持する。保存graphにnative SiLU / Sigmoidが残らずExp / Divを使うこと、他exportの既定分解を変更しないことを含め、encoder / 準備監査の関連32 passed / skip 0（35.78秒）。小さい検証時はOMP / MKL各4 threadとした。elementwise `rtol=1e-4` / `atol=1e-5`と独立した公式encoderの期待値は維持し、実モデルActions再照合を別に行う。

[SiLU修正後CI 38321025453](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38321025453)（PR head `2fb5c83504f46680fd5be40265715ea5eb992eac`、実行merge ref `5b2f387519de6d2029f52d72525fadbe84152c2a`）も不合格。eager対公式差0、保存対eager最大差 `0.00021576881408691406`、許容差比 `9.733773231506348`、不一致368要素。cosine `0.9999997615814209`は合格だが、要素単位の条件は不合格。Softplus / SiLUの小さいgreenを実encoderの合格とはしない。

### 段階別診断のTDD

残る差を推測で修正せず、公式subsample、各encoder層、output projection、最終project済みfeatureを独立したforward hookで取得し、同じ数値経路を使う保存stage graphへ照合する。正常exportは従来どおり単一Tensorを返し、追加graphは数値不合格のときだけ作成する。

- stage export未実装、入力変更後の保存照合、公式hookの例外時解放、不完全 / 非finite trace拒否、最初の差の検出、診断失敗時の元の数値失敗保持について、12件の意図したassertion redを確認。関連44 passed / skip 0（66.43秒）。
- 診断開始前に元の失敗receipt / binaryが存在しない追加1 redを確認。先に元の失敗を保存してから重い診断へ進む最小修正を行った。
- traceの最終値が元の失敗graphと完全一致するかを記録し、一致しない場合も明示する受け入れを追加。診断が実行経路を変えた可能性を隠さず、原因判断の制約として扱う。
- 最終はstage診断14件と既存encoder / 準備32件の計46 passed / skip 0（103.46秒）。`tools/`のuv、OMP / MKL各4 thread、小さいモデルだけで検証。[段階別記録](results/m4-audio-stage-tdd-20261013.json)。

診断では元の `rtol=1e-4` / `atol=1e-5`を維持し、各stageのshape、最大絶対誤差、許容差比、不一致数、最初の不合格stage、診断graphのhashを記録する。公式hookはこの呼び出しが追加したものだけをfinallyで除去する。保存moduleの重複重みを解放してからstage exportを行い、診断例外は別欄へ保持する。元の失敗は引き続きCIを失敗させ、診断成功をencoder / Sentis / 最終768次元の合格にしない。大きいgraphはActions artifactだけに保存する。

[段階別診断CI 38324421389](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38324421389)（PR head `f77139fb5cb7c87ef6cda905fb158e3debd72a0e`、実行merge ref `dc2156c954e66aba418f31f1ce88ca7da089b355`）は数値不合格で完了した。診断自体は実行成功、元の保存graphと診断graphの最終featureは完全一致（最大差0）。subsampleは公式と完全一致し、`layer_0`から小さい差が発生しているが、要素単位許容差を最初に超えるのは`layer_10`（最大差 `0.000148773193359375`、許容差比 `3.572887659072876`、不一致14要素）。`layer_11`は最大差 `0.001194000244140625` / 不一致1146要素、最終projectorでは従来どおり最大差 `0.00021576881408691406` / 不一致368要素だった。[全stageの数値とgraph hash](results/m4-audio-stages-real-failed-20261013.json)。

これで差の蓄積区間は絞れたが、原因演算は未特定である。次は層内のfeed-forward / attention / convolution / normalizationの独立診断をTDDで追加し、最初に差を生む演算を特定する。silenceでgateが停止したため残る5ケースは未検証。Python4環境・lint・packageは合格し、実encoder / Required CIは失敗した。これらの判定を維持して進める。

### 層内診断のTDD

2026-10-13に選択層のfeed_forward1 / norm_pre_attn / self_attn / norm_post_attn / lconv1d / feed_forward2を追加出力する診断を実装した。公式moduleのforward hookから独立に参照を取得し、native側の後続in-place演算前にcloneする。正常exportの出力契約・演算・許容差は維持する。

- 初回の欠落keyword / methodのAPI probeは9件失敗。意図したassertion失敗として確認し直した契約redは10件。最小実装後、保存した変更入力、最終値の元graphとの完全一致、例外時のnested hook解放、不正層selectorを含む関連56件がgreenとなった。
- 最初に許容差超過する層ではなく、最初に非zero差が出た層を選ぶ追加の3 redを確認した。coarse traceが元の失敗graphの最終値と一致するときだけ自動で層内診断する。一致しないときは実行せず、最終経路が変わった制約を保持する。
- coarse診断のJSONを層内export開始前に保存し、追加診断の失敗は別欄へ保持する。元のencoder不合格と粗い診断を消さない。
- 最終は新規13件と既存46件の計59 passed / failed 0 / skipped 0、177.64秒。tools/uv、Python 3.14、OMP / MKL各4 thread、小さいモデルのTDD。[結果とsource hash](results/m4-audio-layer-tdd-20261013.json)。実モデルの原因修正や合格はまだ得ていない。

前回の実重みではsubsampleが完全一致、最初の非zero差はlayer 0なので、次のActionsはlayer 0の層内traceを生成する。数値不合格が残る場合は同じCIを失敗させ、大きいgraphはartifactだけに保存する。診断traceが元のgraphの経路を変えていないことと、どの演算で差が始まるかを先に確認してから、原因の小さい再現テストと修正へ進む。

### 実重みの層内診断

[CI 38327706633](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38327706633)はPR head `0d4b53d8b77bd2213fb3af9683265ac47c727f43` / 実行merge ref `adede1c25500bc05542c4c65f90530eec7a606ee`で数値不合格となった。coarse / 層内traceの最終値はともに元の保存graphと完全一致し、追加診断例外はなかった。[全結果](results/m4-audio-layer-real-failed-20261013.json)。

silenceの公式対eager差は0、保存graph対eager最大差 `0.00013375282287597656`、許容差比 `5.294248580932617`、不一致328要素。前回の368要素とは異なる実run結果として保持し、不合格解消や高速化とはしない。残る5信号は引き続き未実行。

subsampleは一致し、最初の非zero差は`layer_0.feed_forward1`（最大差 `3.814697265625e-6`、許容差比 `0.02367383986711502`）。その後のnorm_pre_attn / self_attn / norm_post_attn / lconv1d / feed_forward2にも差があり、layer 0内では許容差以内だがlayer 10で初めて超過する。feed-forward内のどの演算が最初に差を生むかは未特定。次は公式の入力を固定した小さいfeed-forward単独診断をTDDで作り、RMS norm / linear / activation / residualを個別に照合する。単独traceと全体traceの同じstageの一致を確認してから原因の修正へ進む。

## 次の受け入れ

1. feed_forward1単独診断をTDDで追加し、公式の入力を固定して最初の原因演算を特定する。原因を分離した小さいredから修正し、元のencoder / projector全ケースを再照合する。現時点で実モデル数値不合格は未解消。
2. 音声前処理・encoderのSentis import / 保存・再読込、実CPU / GPUと採用Float16重みの数値照合。非互換・OOM・fallbackは失敗として記録する。
3. A2のPCM入力契約とA3のGPU前処理を接続し、中間CPU readbackなしを確認する。
4. A5のGPU soft token → 共通本体 → 有限768次元・単位長、FP32 cosine >= 0.999 / Float16 >= 0.99、text / image回帰。
5. A6 / A7の全検索順位、再生・明示収録 / 停止・解放、Playerと必須実機を別に検証する。

## Feed-forward単独の演算診断をTDD追加

layer 0.feed_forward1の実入力を公式audio towerのpre-hookから取得し、clamp → pre RMSNorm → linear 1 → activation → linear 2 → clamp → post RMSNorm → scale → residualを比較する小さい診断graphを追加した。独立native hookの出力はin-place更新前にcloneし、失敗時も自分のhookだけを解放する。単独graphのfinalが元のfull traceの同stageと完全一致した場合だけlocalizationを有効とし、一致しなければcontextの違いを残す。

未実装による16 assertion red → 最小実装16 green。数値不合格のないtraceでも余分なexportを始める追加1 redを確認し、実際の数値不合格とfull final一致がある場合だけ子診断を起動するよう修正した。関連encoder / prepare / 段階・層内診断を含め76 passed / fail 0 / skip 0（297.37秒）。[TDD・source / log hash](results/m4-audio-feed-forward-tdd-20261013.json)。

全体graphをさらにもう1つ生成する代わりに、選んだfeed-forwardの重みと公式入力だけでexportする。graph / native inputのsize・hash、各stageの差・固定許容差、standalone対full stage一致をreceiptへ残す。親の不合格報告を子export前に保存し、子diagnosticの例外でも元の数値不合格を保持する。これは診断追加であり、実重みの原因演算・encoder数値合格は未確認。次のActions実行で確認し、既存の不一致328要素と残り5信号未実行を消さない。Unity / end-to-endの受け入れも未完了。

## 実feed-forwardの最初の微差と小さいvector Red

後続CI 38331914586は不合格、producer source `e81255dcbdf9844752bfef0fa789b54635388a47`。silenceの公式対eager差0、保存対eager最大差 `0.00021576881408691406`、許容差比 `9.733773231506348`、368要素不一致。単独feed-forwardはfull traceと完全一致し、clamp・最初のRMSNorm・linearは差0、最初の微差はactivation（最大 `5.960464477539063e-8`）。9 stageは個別許容差内であり、encoder全体の不合格を解消した証拠ではない。

seed 412・float32入力のnative SiLUと保存graphを照合する17要素 / [1,2,4096]の追加2テストは意図したassertion red。不一致は1 / 381要素。実重みを使わない7つの式（exp / expm1 / exp2、sigmoid、float64経由等）を比較したが、どれもnativeとのbitwise一致を満たさなかった。これは原因修正やgreenではなく、採用できない候補を切り分けた小さい調査結果。float64経由の修正や許容差変更は採用していない。[実診断・red・比較とhash](results/m4-audio-activation-vector-red-20261013.json)。

このcheckpointは局所のredを保持するローカルcommitで、未解決のままmainへ統合しない。次はnative vector演算とexport可能な演算の違いを調べ、最小修正後にこのredと既存回帰、元の実重み6信号・Sentis CPU / 実GPUへ進む。診断のbitwise検査と元の固定 `rtol=1e-4` / `atol=1e-5`のencoder受け入れを区別する。

## 固定CPUの数学経路と同一値のvector / scalar差

重み不要の追跡で、実環境はtorch 2.14.1+cpu、AVX2、MKL available、installed AT_MKL_ENABLED=1と確認した。[固定native SiLU](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cpu/Activation.cpp)はvectorでVectorized.exp、scalarでstd::expを使う。installed vec256_float.hのfloat expはSleef_expf8_u10。[固定Tensor.exp](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp)はVML経路、installed vml.hのfloat specializationはMKL Expとなっている。式が同じでも、実行する数学経路は同一ではない。

seed 412の17要素では、index 2 / input -1.0837361812591553についてnative vector=-0.2739681303501129、native scalar=-0.2739681005477905、div_exp=-0.2739681005477905。vector対scalar、vector対div_expは各1要素不一致。同じ入力値でもnativeのarray / scalar条件で丸めが変わることを確認した。[source / probe hashと範囲](results/m4-audio-native-math-research-20261013.json)。

これは保存式とnativeの経路差を絞った研究結果。全体encoder不合格の厳密な因果、export可能な修正式、Sentis encoderの成功は未証明。referenceの数学backendを変更して合格にせず、許容差・本番実装は変更していない。局所redを保持し、次の候補はscalar式の一致だけで選ばずvector条件と元の実重み6信号へ照合する。
