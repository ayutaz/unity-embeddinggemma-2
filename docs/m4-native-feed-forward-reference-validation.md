# 固定native layer 0のfeed_forward1内部参照

> 文書整合確認: 2026-10-14。[現在の状態](status.md) / [GPUを主経路とする再開計画](multimodal-tdd-plan.md)。本文の数値・source・実行中／次の作業は記載当時の記録。現在headの実行結果として扱わない。

2026-10-14 JST。Sentisの固定silence入力ではnorm_pre_attn入力時点で既に数値条件を超えるため、同じnative文脈のfeed_forward1内部5段階を取得する。Runtime・モデル・元参照・許容差を変更せず、正式Release / tagは作成しない。

## TDD

詳細capture不足の3 assertion red、producerの詳細flag不足1 TypeError red、CLI flag不足1 argparse redを確認した。詳細capture / producer / CLIを実装し、既存14件を含む19 green。追加workflow条件の1 assertion red → 1 greenも確認した。ローカルでは小さい合成fixtureのみ、tools/とuvを使用し、実weightの取得・実モデル実行は行っていない。

公式Gemma4AudioLayerのfeed_forward1内のpre_layer_norm / ffw_layer_1 / act_fn / ffw_layer_2 / post_layer_normへ自身のhookを登録する。入力をcloneし、出力はその場でcloneする。post_layer_normの値は後続のscale / residualのin-place更新前に保存し、独立した公式module計算へ完全一致することをテストした。例外でも自身のhookだけを削除し、外部hookを保持する。固定native layer finalとの完全一致を満たさない実行は拒否する。

`--detail-feed-forward`時も公式layer 0の実行は1回。元7段階に5段階を追加し、元7ファイルのshape / size / SHAを変えない。通常モードは従来の7段階を保持する。新しい内部値は採用前に同一producerの最終値で文脈一致を確認する。

専用Actionsで元artifact `11814567883`から入力・最終値と固定manifestのみrange取得し、SHA監査後にconfig / safetensorsを取得する。成功でもfull_encoder / Unity受け入れにはしない。実モデルの詳細参照はまだ未取得であり、取得後にSentis保存モデルの同段階へ係数identityと演算経路を使って対応付ける。

元の音声ONNX silence不合格、Sentis encoder全12条件の不合格、画像FP32条件、全chain / role・境界、sample・実モデルPlayer / 必須実機・最終配布準備は未完了。

## 実モデル結果

[Draft PR #77](https://github.com/ayutaz/unity-embeddinggemma-2/pull/77)、head `41e3199ca1816751a8d5a968d923aca1b7700b7b`の[run 38463246855](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38463246855)が成功。実行merge source `6ac2c92191a62ada4830adeee5235fb3c4b90f15`、report SHA `99529dc0…`を固定した。[小さいreceipt](results/m4-native-feed-forward-replay-actions-20261014.json)。元layer finalは全要素bit一致し、元7ファイルのSHAも以前のrunと全一致した。

追加5出力はpre norm / linear2 / post normが`[1,280,1024]`、linear1 / activationが`[1,280,4096]`。モデルweightは取得先Actionsで使用し、成果物へ含めない。ダウンロード後の全12 binaryでsize / SHAを再照合した。この成功はnative内部参照の同一文脈再現であり、元音声ONNX・Sentis数値不合格を解消したものではない。通常CIは別runとして追跡する。

同じheadの[通常CI 38463244810](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38463244810)はPython4環境 / package / actionlintに合格したが、model-referenceは元の`audio ONNX export mismatch: silence`で失敗しRequired CIも不合格。これをnative参照取得成功で置き換えない。診断完了後は専用labelを外し、文書だけの更新で同じ参照を重複実行しない。
