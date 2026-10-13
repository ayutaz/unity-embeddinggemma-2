# M3: 実画像から最終embeddingへのPython参照

2026-10-13。I1 / I4へ渡す最終vector参照と、実feature数に対応するbackbone profileをTDDで追加した。これは画像Runtime API / sampleやSentis GPU連結の完了ではない。既知の画像FP32・音声encoder不合格、元の全12条件、最終空consumer・Player / 他OSのgateを維持し、正式Release / tagを作成しない。

## 参照と保存経路

`load_image_embedding_model`は公式`EmbeddingGemma2Model`をmeta構造で作り、実text model、vision tower、vision projectorを既存のstrict loaderでFloat32へ読み込む。画像の未使用audio towerは構築・読み込みしない。公式forwardを差し替えず、pixel / position IDを渡して、画像feature配置とlanguage model出力を得る。既存の固定参照と同じpromptを含むmasked mean・L2正規化を最終参照にする。unused audioを省くことをメモリの実測改善とは表現しない。

`prepare-image-embeddings`は同じproducer / revisionのvision参照と全file hashを読み戻し、モデル・processorのsnapshot hashも照合する。画像256 / 252 feature等に公式markerを展開し、document prefixと完全なtoken / maskを使う。長さを切り捨てない。backboneは画像の実feature数とaudio 0を持つ4入力profileをexportする。合成280 / 280のgraphを流用しない。

各画像で、公式全forward、保存backboneへnative画像featureを渡す経路、保存vision encoder → 保存backboneの経路を比較する。featureは `rtol=1e-4` / `atol=1e-5`、最終vectorも同じelementwise許容差とcosine >= 0.999999を要求する。native featureだけの比較で保存encoderの破損を見逃さない。全3 profile / 6画像を保持し、同じbackbone shapeのlandscape / portraitは1つの保存graphを共有する。小さいfixtureでは3 profile / 6画像で同shapeのgraphが3個から1個になり、全画像が照合された。実モデルでの保存容量・時間・memoryは未測定。

## TDD

- loader、公式最終vector・保存profile、source / revision / hash gate、CLIの未実装を8 assertion red → 8 green。既存loader・vision・backbone・promptを含む68 green。
- graph共有と未監査pixel / snapshot hash名の拒否を3 red、position参照・出力case名の拒否を2 redから実装し、関連73 green。
- 保存encoderのprojectionを壊しても成功する穴と重複case IDを2 redで確認し、保存encoderを含む最終chain照合と早期拒否を追加した。最終関連75 passed / failed 0。小さいモデルの生成weightsを使うfixtureであり、実重み / GPU互換性の証拠ではない。

[TDD receipt](results/m3-image-embedding-reference-tdd-20261013.json)。ローカルで実モデルを生成・実行していない。Pythonはtools/とuvを使った。

## 実モデルと次の受け入れ

Actionsの既存vision生成後に実行し、固定revisionの重みを同じrunner cacheから利用する。最終参照JSON、profile graph、native feature binary、全hashをartifactへ保存し、モデルをGitへ含めない。`python_saved_chain_validated`、`unity_validated`、`gpu_chain_validated`を分けて記録する。全graphを1度に保持せず、現在のprofileを処理して解放する。

実モデルPython結果は未実行。次はActionsの結果とsourceを確認し、実CPU / GPUの対応profileとencoderから最終768次元までをSentisで照合する。zero-audio inputのimport / Runtime受け入れ、画像FP32の元gate、GPU前処理の中間readback禁止と資源・性能を別々に確認する。失敗や未実行を成功にしない。
