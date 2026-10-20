# ドキュメント

更新: **2026-10-21**。Windows版 **[v0.1.0は正式公開済み](https://github.com/ayutaz/unity-embeddinggemma-2/releases/tag/v0.1.0)** です。
初めて使う場合は[READMEの導入手順](../README.md#導入)から始めてください。
現状は[検証状況](status.md)、これからの作業は[ロードマップ](roadmap.md)へ集約しています。

## 利用する

| 文書 | 内容 |
| --- | --- |
| [導入・Runtime API](../Packages/com.ayutaz.embeddinggemma/Documentation~/index.md) | UPM導入、C#の使用例、所有権と破棄 |
| [モデル準備](model-preparation.md) / [公開モデルpreview](media-model-preparation.md) | text単体の変換と、変換済み全6 moduleの明示取得・hash照合 |
| [Text Search](../Packages/com.ayutaz.embeddinggemma/Samples~/TextSearch/README.md) | テキスト検索sample |
| [Image and Audio Search](../Packages/com.ayutaz.embeddinggemma/Samples~/MultimodalSearch/README.md) | 画像・音声、入力profile、検索、cancel / recovery |
| [API詳細](runtime-api.md) / [マルチモーダルAPI](../Packages/com.ayutaz.embeddinggemma/Documentation~/multimodal-api.md) | GPU設定、前処理・encoder・backboneと寿命 |
| [音声API](../Packages/com.ayutaz.embeddinggemma/Documentation~/audio-embedder.md) | 非同期呼び出しとstageの所有権 |
| [ライセンス](model-license.md) | SDKと別取得モデルの利用条件 |

## 現状・計画・開発

| 文書 | 内容 |
| --- | --- |
| [検証状況](status.md) / [正式版の確認記録](results/windows-v010-release-20261021.json) | 公開tag、実Editor / Player、数値、既知の限界 |
| [対応構成](windows-supported-configuration.md) | 検証したWindows・GPU・メモリと非保証範囲 |
| [ロードマップ](roadmap.md) / [ゴール](goal.md) | 次の優先順・開始条件と維持するGPU / 精度要件 |
| [CI](ci.md) / [Unity自動操作](automation.md) | 開発手順、workflow、実行監督と結果の判定 |
| [privacy](privacy.md) / [モデルmetadata検査](model-artifact-metadata.md) | Git hooks、履歴・payload・画像・モデル内部文字列の検査 |
| [端末検証](device-validation.md) / [混在Playerの端末経路](device-mixed-runtime.md) | 将来のmacOS / Android / iOS受け入れ条件 |

## 性能・参照・配布

- [Windows画像・音声のGPU性能とメモリ](windows-media-memory-results.md)
- [テキストのCPU / GPU比較](text-paired-performance.md)
- [CPU参照の測定契約](reference-cpu-performance.md)、[画像の固定条件](image-CPU-fixed-conditions.md)、[音声CPU結果](native-audio-CPU-results.md)
- [Windows CPU CIとhost差](windows-CPU-CI.md)、[旧CPU診断](reference-cpu-diagnostics.md)、[ONNX CPU比較](onnx-cpu-comparison.md)
- [native回帰データ](native-regression-distribution.md)、[元画像参照JSON](image-reference-distribution.md)、[旧raw候補の扱い](raw-candidate-archive.md)

各結果のsource・入力・機器・計測範囲を維持します。異なるhostの時間を同じ機器の速度比較へ合算しません。

## 固定sourceの検証記録

以下は記載当時の成功・失敗・未実行の記録です。本文の「現在」「次の作業」「未統合」は当時の状態を指し、現在の作業キューには使いません。

| 分野 | 入口 |
| --- | --- |
| 正式公開前のWindows候補 | [工程1〜3の記録](results/windows-first-stages-1-to-3-20261020.json)、[従来の混在Player](results/final-mixed-player-3a02fc0-20261019.json)、[従来text Player](results/final-text-player-3a02fc0-20261019.json) |
| サンプルと実操作 | [画像・音声sample](multimodal-search-sample.md)、[音声の途中停止](results/native-playback-stop-7754ce4-20261019.json) |
| 共通Runtime | [backbone](multimodal-backbone-validation.md)、[prompt](multimodal-prompt-validation.md)、[GPU constant / IL2CPP](gpu-constant-il2cpp-validation.md)、[GPU出力確保](gpu-write-only-output-allocation.md) |
| テキスト | [M1完了](m1-completion-validation.md)、[Runtime](m1-runtime-validation.md)、[基盤](m1-validation.md) |
| UPM・consumer・Player | [package](m2-package-validation.md)、[固定Git consumer](m2-latest-git-consumer-validation.md)、[検索](m2-search-validation.md)、[IL2CPP Player](m2-player-il2cpp-validation.md)、[sample Player](m2-player-sample-validation.md) |
| 画像 | [前処理](m3-image-preprocessing-validation.md)、[encoder](m3-image-feature-encoder-validation.md)、[最終API](m3-image-embedder-validation.md)、[元参照](m3-image-embedding-reference-validation.md) |
| 音声 | [前処理](m4-audio-preprocessing-sentis-validation.md)、[encoder](m4-audio-encoder-runtime-validation.md)、[最終API](m4-audio-embedder-validation.md)、[元参照](m4-audio-final-reference-validation.md)、[旧診断ledger](steps-1-3-acceptance.md) |
| 計測手法 | [GPU submission](media-gpu-submission-observations.md)、[load / release](load-release-measurement-scope.md)、[GPU memory](windows-gpu-memory-observation.md)、[GPU engine](windows-gpu-engine-observations.md)、[async frame GC](async-frame-allocation-observations.md) |
| 調査・設計 | [技術調査](technical-approach.md)、[新規性調査](embeddinggemma-2-unity-novelty.md) |

`results/`のJSONと、関連する固定sourceの検証・設計・計測資料は保持しています。古い未実行手順を重ねた計画書は削除し、
今後の実行順は[ロードマップ](roadmap.md)だけを更新します。重み・生log・個人の録音・大きい生成artifactはGitへ含めません。
