# M3画像encoder / projectorの固定profile export

更新日: 2026-10-13。[M3計画](m3-plan.md)のI1 / I2を進める。画像end-to-end API / sample / GPU前処理・Sentis実モデルは未完了。**この段階の出力は共通本体へ渡す512次元のsoft tokenであり、最終768次元embeddingではない**。

## 実装とTDD

- 公式Gemma4VisionModelとEmbeddingGemma2MultimodalEmbedderだけをFP32でstrict loadする。text / audio encoderは構築・ロードしない。projection欠落は拒否する。
- 固定batch 1、RGB patch、位置profileを使う。公式patch embedding / axial RoPE / attention / 層 / projectorを維持し、動的なboolean padding選択を静的なpooling行列へ置き換える。3×3空間pooling、sqrt(hidden_size) scaling、standardizeを公式出力と照合した。
- profileは完全なrow-major矩形と末尾padding。位置table範囲、poolingの整除、pixel形状・FP32・有限値・[0,1]を検査する。export済み数値graphにはhost入力検査を含めないため、Unity側に同じ検査が必要。
- 未実装export / loaderの5 red、位置・pixel・standardizeを含む21 redを記録。参照生成 / 保存済み誤出力拒否 / stale成功除去 / CLI / 必須workflowは5 red。不正profile事前拒否4 red、公式processorと同じTensor形式のtoken数の処理は1 redを確認して修正した。
- 最終の関連回帰は66 passed / skipped 0。小さいvision modelと公式processorで実行した。公式processorのtoken数をlistと仮定したテストfixtureの失敗は修正履歴として残し、アルゴリズムred件数に数えない。
- 公式processorを使う参照生成のため`uv add --no-sync torchvision`で依存とlockを更新。実モデル処理はActionsへ寄せ、ローカルでは小さい契約テストだけを実行した。

## 必須Actions gate

```powershell
cd tools
uv run --locked python -m embeddinggemma_tools prepare-vision --output ../artifacts/vision-encoder
```

固定model revision `914f7f89142e33e77833254d9c9b90c3cef7303b`のprocessor設定と実vision / projector重みを使う。公式processorのstandard budget 280で正方形・横長・縦長の3 profile、profileごとに異なる生成RGB画像2ケースを扱う。原画像は96×96 / 96×160 / 160×96、実入力寸法・patch数・soft token数は公式resize結果から決める。280を実際の出力数とみなさず、正方形は256 tokenになることを小さい前処理fixtureで確認した。

各profileの保存Core ATen `.pt2`を再ロードし、公式encoder / projector参照の全soft tokenに対してcosine >= 0.999999、elementwise rtol 1e-4 / atol 1e-5、有限値・shape一致を要求する。単純なimport成功やcosineだけで合格にしない。

`vision-reference.json`にprofile、各ケースの公式feature、source / model / processor hash、lockされたlibrary版と先に決めた許容差を記録する。原画像PNG、RGB patchのlittle-endian FP32 binary、位置int32 binary、graph・manifestを3日保持の別artifactへ置く。入力画像はコードで生成したpatternであり、ユーザー入力や外部の写真を収集しない。

成功報告の`stage=vision_encoder_projector_only`、`unity_validated=false`、`end_to_end_validated=false`を保持する。失敗時は前回の成功receiptを除去し、生成済み部分artifactと失敗logを残す。実Actions結果は取得後に追記する。

## 実重みActionsの結果

[PR #37 CI 38310718250](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38310718250)は全8 job成功。実visionの3 profile / 6ケース、minimum cosine `0.9999997615814209`、elementwise条件・有限値 / shapeが合格した。参照 / artifact sourceは`e2a6dc01fbca8c9be1f3bbe875325ad338de0ca0`（PR head `b4222c564de415cbf09a39864a832ef8bd839d9b`のmerge ref）。統合後main `08f3dfc406738943a54e773d401dc502557f6f04`の[CI 38311759146](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38311759146)も成功。Python参照の成功であり、Sentis・GPU前処理・最終embedding・M3全体の合格ではない。

## 次の検証

1. 上記の合格artifactを監査・再利用し、非対応演算・shape・memory不足は失敗として保存する。
2. Sentis import / 保存・再読込、実CPU / GPUのfeature照合とFloat16重み検証。
3. I3のGPU resize / RGB patch化。公式前処理の中間tensorを参照に使い、向き・色空間・alpha・奇数寸法・縦横比をTDDで確認する。
4. imageの実feature数に一致するbackbone profile、入力 / GPU資源契約、最終768次元・テキスト回帰、検索sample / Player。

意味検索の画像・日英query / 全順位はI5で別fixtureへ固定する。pattern画像のencoder照合を意味品質・実GPU・他OSの合格へ流用しない。正式リリースは行わない。
