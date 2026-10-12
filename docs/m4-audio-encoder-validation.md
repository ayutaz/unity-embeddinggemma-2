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

この文書追加時点では、小さいfixtureの合格のみ取得済み。実encoderのActions結果は実行後に追記し、Python export成功をSentis / GPU / 最終vectorの成功には数えない。

## 次の受け入れ

1. Actionsの固定実重みによるencoder / projector照合。実profileとtoken数・graph hashを取得する。
2. 音声前処理・encoderのSentis import / 保存・再読込、実CPU / GPUと採用Float16重みの数値照合。非互換・OOM・fallbackは失敗として記録する。
3. A2のPCM入力契約とA3のGPU前処理を接続し、中間CPU readbackなしを確認する。
4. A5のGPU soft token → 共通本体 → 有限768次元・単位長、FP32 cosine >= 0.999 / Float16 >= 0.99、text / image回帰。
5. A6 / A7の全検索順位、再生・明示収録 / 停止・解放、Playerと必須実機を別に検証する。
