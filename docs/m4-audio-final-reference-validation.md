# 独立した最終音声Python参照

2026-10-13。公式`EmbeddingGemma2Model`のaudio-only forwardから、元6信号 / 2 profileのDocument-role最終vectorを生成するCLIをTDDで追加した。画像towerをmeta構造にも含めず、使用するtext / audio / projectorだけをstrict loadする。係数・mask・encoder feature・保存native encoder→保存backboneと、公式最終vectorを同じ固定許容差へ照合する。画像feature数0、音声280 / 210 soft tokenの実backbone profileはActionsで生成する。

TDDはloader / 最終参照の7 assertion red → 7 green、CLI / workflowの2 red → 既存stageを含む4 green。元ONNX失敗状態が最終報告へ渡らない1 redを修正し、最終関連11 passed / failed 0。[小さいTDD receipt](results/m4-audio-final-reference-tdd-20261013.json)。これは実重み・Unityの成功とは別。

参照JSONのhashとsnapshot / sourceを重み読み込み前に監査する。全6 ID / 2 profile、非empty prefix mask、finite feature、zero paddingを確認する。公式audio featureは元encoder参照へ照合し、最終vectorは公式forwardのtoken単位投影→mask付きmean→L2 normalizeから独立に求める。保存backboneに公式featureを与える比較と、保存native encoderを含むchainの比較を別々に行う。

native `.pt2`のPython chain合格は標準ONNX encoderやSentisの合格を意味しない。元encoderのsuccess・ONNX実行件数 / 失敗件数・数値合格を保持し、`python_native_saved_chain_validated`、`onnx_encoder_numerically_validated`、`unity_validated`、`gpu_chain_validated`、`end_to_end_validated`を分ける。旧固定参照の不合格と参照driftも上書きしない。

```powershell
cd tools
uv run --locked python -m embeddinggemma_tools prepare-audio-embeddings --encoder-source ../artifacts/audio-runtime-reference --output ../artifacts/audio-final-reference
```

この実重み生成は同repoのopt-in `audio-runtime-diagnostics` workflowで行い、native参照完了manifestがある場合だけalways条件で続行する。collectionのONNX不合格は非zero終了のまま、workflow failureを維持する。必須model-referenceの厳格なgateは変更しない。最終backbone / JSON / native featuresは別artifactへ保存し、Gitへ重み・生成モデルを入れない。

次は実重みの独立参照生成、source / hash / maskの監査、音声専用profileのSentis import・CPU / 実GPUとRuntime APIの全chain照合。Raw / Query-roleの追加参照、検索 / 再生 / 明示収録sample、Player・各必須実機・最終空consumerは別gateで残る。Document-role参照や小さい11 greenを全対応の完了にしない。

### 実重み生成後のcheckpoint

[Actions run 38398829258](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38398829258)で最終参照生成とuploadが成功した。producer `4edbe17a8105233c5aaa1514faa2862b08a4f7e0`とhead `87f2b4dc3dcf4d6618120634b2d400d985871e40`を区別する。6信号 / 2 profile、全768次元、image count=0、audio count=280 / 210、sequence length=1024、feature width=512をJSONへ確認した。保存backboneとnative保存encoder→保存backboneは固定elementwise条件とcosine >= 0.999999を満たし、minimum cosineは0.9999999403953552。[実参照receipt](results/m4-audio-final-reference-real-20261013.json)。浮動小数のcosineには1をわずかに超える値もあり、未加工の値を参照JSONへ保持する。

最終artifactは2,194,146,655 bytes。306,255 bytes / 9 requestでJSON2つだけを範囲取得し、CRC・reference SHAとmanifest、metadata一致を監査した。backbone binariesは取得・Unity importしていない。元encoderのsuccess=false / ONNX全6件不合格を保持し、workflow全体はfailure。Required CI run 38398824940もmodel-reference / Required CIがfailureで、Python全4環境・workflow lint・packageは成功した。native最終参照の合格をONNX / Sentis / 旧固定参照の問題解消へ流用しない。生成labelは解除済み。

次はこの新しい最終参照sourceを明示した音声専用backboneのSentis単独受け入れと、Runtime APIを使った全chain照合。同時に元FP32 encoderと固定参照driftの原因修正を追う。全chainが合格しても、未達のencoder条件を消さず、M3 / M4・各role / sample・必須環境・配布準備まで元のscopeを維持する。

### 実backbone受け入れハーネスのTDD

`eeec732`で、大きいartifactのbounded streaming / 最大4並列range / CRC検証済みcache再利用と、実音声専用backboneの受け入れハーネスを追加した。取得処理は意図した3 red / 既存1 passから関連15 green。Unity guardはsource・全6信号・profile・非finite出力・既存FP32 / Float16最終cosine条件の9 assertion redから9 green、compile error / warning 0。固定Git Runtime `9c6f794…`のconsumerで、受け入れtestのみを配置しRuntime overlayはない。[TDD receipt](results/m4-audio-final-backbone-harness-tdd-20261013.json)。

実backbone試験は新参照producer `4edbe17…`のnative features → image count=0 / audio count=280または210のbackboneを全6信号 × CPU / GPU × FP32 / Float16で照合する。profileごとのimport / 保存は1回にし、各model / workerを解放して次へ進む。全caseの数値結果・actual backend・fallback / flushを保存する。最終vectorの既存cosine条件と診断用elementwise値を分け、encoderの固定elementwise gateは変更しない。このTDD checkpointでは実モデル試験は未実行。取得のHTTP 403はインフラ失敗として別記し、取得済みmemberを検証・再利用して続行する。
