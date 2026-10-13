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
