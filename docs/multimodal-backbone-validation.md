# 共通マルチモーダルbackboneの準備と検証

> 設計・検証の履歴です。本文の状態・実行中・次の作業は記載当時のものです。現在の対応範囲は[検証状況](status.md)、残作業は[ロードマップ](roadmap.md)を参照してください。固定sourceの失敗・未実行も証拠として保持します。

更新日: 2026-10-13。[全体計画](multimodal-tdd-plan.md)のC2 / C3の一部を実装する。ユーザーから工程6までの実行を依頼されたが、**正式リリースは今回行わない**。最終配布準備まで進め、公開操作は保留する。

## 実装とTDD

- 特徴量入力の14件で未実装のassertion失敗を確認した後、画像 / 音声のproject済みsoft tokenを受け取るbatch 1の共通backbone exportを追加した。既存export / model回帰を含め18 passed。
- profile不正設定・backbone参照生成 / stale成功除去は8 failed → green、CLI / 必須workflow接続は2 failed → green。最終の関連契約 / 回帰は43 passed、skip 0。ローカルは既存uv環境の小さいモデルだけを使用し、実重みの取得 / 実行はActionsへ寄せた。
- 数値の独立参照は公式multimodalモデルと同じsafe ID置換 / masked-scatterを使い、exportは累積slot番号・gather・Whereへ置き換える。非連続placeholder、特徴量の並び変更、padding、OOV modality ID、単一modality・text-only、局所注意窓をまたぐ長さを照合する。
- batch / 長さ / feature数 / 幅、binary mask、padding中のslot、語彙外text ID、非有限値 / 型 / deviceをeager入力で拒否する。**export済み数値グラフにはhost側の入力監査を含めない**ため、Unity側も同じprofile契約を実装してから公開APIへ接続する。

## Actionsの実重みgate

必須`model-reference` jobに次を追加する。既存text128の参照 / export照合を維持し、別artifactとscopeで記録する。

```powershell
cd tools
uv run --locked python -m embeddinggemma_tools prepare-backbone --output ../artifacts/multimodal-backbone
```

- 固定revision `914f7f89142e33e77833254d9c9b90c3cef7303b`、float32の実text backbone、batch 1 / 長さ1024、画像280 / 音声280のproject済み特徴量。これはsyntheticな特徴量入力の互換性fixtureであり、実画像 / 音声のencoder出力ではない。
- modalityの順序とfeature値を変更した2ケースを、独立Python参照と保存済みCore ATen programへ渡す。cosine >= 0.999999と有限値を要求し、失敗時は成功報告を保存しない。
- `backbone-reference.json`に入力 / feature / vector、source SHA、model revision、config / processor / weights hash、library版を保存する。`backbone-validation.json`と各成果物hashを監査証拠にする。
- `stage=projected_features_backbone_only`、`encoders_validated=false`、`unity_validated=false`を必ず記録する。CIの成功をM3 / M4やSentis GPU成功へ読み替えない。
- 生成graph・重み・大きい参照はGit / LFSへ入れず、3日保持のActions artifactへ置く。

[PR #34 CI 38306145598](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38306145598)のログで、`success=true`、2ケース、minimum cosine 1.0、`stage=projected_features_backbone_only`、`unity_validated=false`、`encoders_validated=false`を確認した。統合後main `0ea29ac5dee26f201dd1d2ba30e78221acdbe6d3`の[CI 38306673212](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38306673212)も成功。実text重みのPython export合格であり、Sentisや実encoderの合格ではない。

## 未完了

1. 新1024長exportのSentis import / 実CPU / GPU数値照合。実text重みのActions export照合は上記で合格。
2. 公式processorの画像 / 音声前処理とtoken数契約の固定、encoderとprojectorの個別export / 実モデル照合。
3. Unity側のGPU tensor受け渡しと入力profile監査、画像 / 音声API・sample・Player・全必須環境。
4. 標準21f1 compile・新固定Git candidate導入、Float16/GPU性能・資源解放のM2残作業。
5. 全必須条件の合格後の最終配布準備。正式リリースは行わない。
