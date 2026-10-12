# M4音声encoder / projectorの固定profile export

更新日: 2026-10-13。[M4計画](m4-plan.md)のA1 / A4を進める。**この出力は共通本体へ渡す512次元soft tokenであり、最終768次元embeddingではない**。Sentis実CPU / GPU、共通本体・検索 / 再生 / マイク、各実機は未検証。正式リリースは作らない。

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

## 次の受け入れ

1. Actionsの固定実重みで段階別診断を実行し、最初に差が出るstageを特定する。原因を分離した小さいredから修正し、元のencoder / projector全ケースを再照合する。現時点で実モデル数値不合格は未解消。
2. 音声前処理・encoderのSentis import / 保存・再読込、実CPU / GPUと採用Float16重みの数値照合。非互換・OOM・fallbackは失敗として記録する。
3. A2のPCM入力契約とA3のGPU前処理を接続し、中間CPU readbackなしを確認する。
4. A5のGPU soft token → 共通本体 → 有限768次元・単位長、FP32 cosine >= 0.999 / Float16 >= 0.99、text / image回帰。
5. A6 / A7の全検索順位、再生・明示収録 / 停止・解放、Playerと必須実機を別に検証する。
