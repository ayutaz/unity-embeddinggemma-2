# ドキュメント

初めて利用する場合は[リポジトリのREADME](../README.md)から始めてください。現在の事実は[検証状況](status.md)、これからの作業は[ロードマップ](roadmap.md)を基準にします。

## 利用する

| 文書 | 内容 |
| --- | --- |
| [導入・Runtime API](../Packages/com.ayutaz.embeddinggemma/Documentation~/index.md) | UPM導入、C#の使用例、所有権と破棄 |
| [モデル準備](model-preparation.md) | テキストモデルの取得・変換、画像・音声bundleの制約 |
| [Text Search](../Packages/com.ayutaz.embeddinggemma/Samples~/TextSearch/README.md) | テキスト検索サンプルの操作 |
| [Image and Audio Search](../Packages/com.ayutaz.embeddinggemma/Samples~/MultimodalSearch/README.md) | 画像・音声の登録、入力profile、検索・キャンセル |
| [API詳細](runtime-api.md) | GPU設定、前処理、encoder・backbone、保存API |
| [低レベルのマルチモーダルAPI](../Packages/com.ayutaz.embeddinggemma/Documentation~/multimodal-api.md) | prompt・GPU特徴量・前処理の入力と所有権 |
| [音声API](../Packages/com.ayutaz.embeddinggemma/Documentation~/audio-embedder.md) | 非同期呼び出し・stageの所有権 |
| [モデルのライセンス](model-license.md) | コードとモデルの利用条件・配布境界 |

## 現状と開発計画

| 文書 | 内容 |
| --- | --- |
| [検証状況](status.md) | 2026-10-18時点の統合済み機能・確認済み環境・未達条件 |
| [ロードマップ](roadmap.md) | 次のPR単位、依存関係、受け入れと正式公開の条件 |
| [プロジェクトのゴール](goal.md) | GPU要件、精度・対応環境、対象外 |
| [CI](ci.md) / [自動操作](automation.md) | 開発環境・workflow・実行と結果の扱い |

## 検証記録と設計資料

以下は固定source・記載当時の実行条件に対応する資料です。本文中の「現在」「次に実行」「未統合」「失敗中」はその時点の記録であり、現在の対応表や作業キューには使いません。失敗と未実行の証拠も保持しています。

| 分野 | 入口 |
| --- | --- |
| 最新サンプルの実GPU・メモリ・画面・回帰 | [画像・音声サンプル検証](multimodal-search-sample.md) |
| GPU出力確保の改善と新SDKの精度・順位 | [CPU storageの省略](gpu-write-only-output-allocation.md) |
| 新SDKの全画像・音声profileのGPU/CPU/wall・メモリ | [同じsourceの40反復](media-gpu-submission-observations.md) |
| 同じSDKのtextのGPU/CPU/wall・メモリ | [両形式・固定queryの4反復](text-current-sdk-observations.md) |
| Editor検索の校正済み割り当て・空/再利用Library | [Editorの別測定](editor-search-allocation-observations.md) |
| awaitをまたぐフレーム全体のGC counter | [既知payloadのcontrolと代表実音声の検証区間](async-frame-allocation-observations.md) |
| Windows IL2CPP Playerの固定source受け入れ | [Player結果JSON](results/m2-current-main-player-acceptance-20261017.json) |
| 元画像・音声encoderと全role API | [GPU/API完了監査](results/goal4-current-completion-audit-20261016.json) / [GPU診断](gpu-execution-and-silu-validation.md) |
| テキストM1の保存・量子化・測定 | [M1完了検証](m1-completion-validation.md) |
| 過去の固定Git consumerと検索 | [Git consumer](m2-latest-git-consumer-validation.md) / [検索検証](m2-search-validation.md) |
| 実装時のTDD分割・checkpoint | [全体TDD設計](multimodal-tdd-plan.md) / [M2](m2-plan.md) / [M3](m3-plan.md) / [M4](m4-plan.md) |
| 正式公開の詳細チェック | [公開準備の設計](m2-release-plan.md) |
| 調査・モデル構造 | [技術方針](technical-approach.md) / [新規性調査](embeddinggemma-2-unity-novelty.md) |

`results/`のJSONには実行source・参照・数値・失敗・測定条件を記録しています。モデル重み・大きい中間テンソルは含みません。CIの成功だけでSentis実GPUや全端末の合格を示すことはできません。

<details>
<summary>検証記録の全一覧（分野別）</summary>

### 共通基盤・GPU

- [GPU constantのIL2CPP保持・実行検証](gpu-constant-il2cpp-validation.md)
- [大きいGPU constantの分割転送](gpu-constant-upload-validation.md)
- [GPU負荷計測とSiLU修正後の実モデル照合](gpu-execution-and-silu-validation.md)
- [共通マルチモーダルbackboneの準備と検証](multimodal-backbone-validation.md)
- [公式マルチモーダルpromptのRuntime契約](multimodal-prompt-validation.md)
- [共通マルチモーダル本体のRuntime契約](multimodal-runtime-validation.md)
- [Player / 他環境の検証準備](player-validation.md)

### M1 テキスト

- [M1 保存・量子化・測定の検証記録](m1-completion-validation.md)
- [実モデル検証・ランタイム API の実行記録](m1-runtime-validation.md)
- [M1 検証記録](m1-validation.md)

### M2 導入・cache・Player

- [Android検証APKのbuild・監査と実機gate](m2-android-validation.md)
- [Player bundleのcache再利用と完全SHA-256高速化](m2-bundle-cache-validation.md)
- [Windows Editorのsample font寿命の修正](m2-editor-font-lifetime-validation.md)
- [Editor待ち時間の切り分け: 2026-10-12](m2-editor-profile-validation.md)
- [Unity21の空Git consumerとembedded Test Framework検証](m2-embedded-framework-validation.md)
- [font修正後の固定Git導入と測定fixtureの寿命](m2-font-git-consumer-validation.md)
- [commit固定Git URL consumerの実動作確認](m2-git-consumer-runtime-validation.md)
- [最新sampleを含む固定Git consumerの検証](m2-latest-git-consumer-validation.md)
- [CI・main保護・UPM移行の検証](m2-package-validation.md)
- [Windows IL2CPP release Playerの検証](m2-player-il2cpp-validation.md)
- [Windows sample Player の検証](m2-player-sample-validation.md)
- [TextSearch sample: 選択モデルの配置・cache検証](m2-sample-cache-validation.md)
- [検索サンプル実装の検証記録](m2-search-validation.md)
- [Windows Editorの起動環境とモデルなし安定性の切り分け](m2-stability-environment-validation.md)
- [テキスト検索サンプルの文字欠け修正](m2-ui-font-validation.md)
- [Unity 6000.3.21f1更新と終了比較](m2-unity-21-upgrade-validation.md)

### M3 画像

- [CPU画像前処理と実測・未達条件](m3-image-cpu-preprocessing-validation.md)
- [最終画像embedding APIの合成受け入れ](m3-image-embedder-validation.md)
- [M3: 実画像から最終embeddingへのPython参照](m3-image-embedding-reference-validation.md)
- [固定画像profileのencoder Runtime API](m3-image-feature-encoder-validation.md)
- [M3: GPU画像前処理のTDDとWindows検証](m3-image-preprocessing-validation.md)
- [M3画像encoder / projectorの固定profile export](m3-vision-export-validation.md)

### M4 音声

- [最終音声APIのTDDと合成検証](m4-audio-embedder-validation.md)
- [音声encoder Runtime接続のTDD](m4-audio-encoder-runtime-validation.md)
- [M4音声encoder / projectorの固定profile export](m4-audio-encoder-validation.md)
- [独立した最終音声Python参照](m4-audio-final-reference-validation.md)
- [M4: native音声参照と標準ONNX Swish候補](m4-audio-native-onnx-validation.md)
- [M4: PCM前処理RuntimeとGPU出力所有権](m4-audio-preprocessing-runtime-validation.md)
- [M4音声前処理のSentis CPU / GPU受け入れ](m4-audio-preprocessing-sentis-validation.md)
- [M4音声前処理の固定profile exportと参照](m4-audio-preprocessing-validation.md)
- [実PCMから最終音声APIへの候補chain検証](m4-audio-real-chain-validation.md)
- [現在の保存encoderの層別数値差](m4-current-coefficient-bound-layer-validation.md)
- [現在の音声encoder数値gateと保存モデルの途中観測](m4-current-encoder-observation-validation.md)
- [layer 0 feed_forward1の保存モデル数値照合](m4-feed-forward-stage-observation-validation.md)
- [固定native入力によるFP64 projection比較](m4-fp64-projection-oracle-validation.md)
- [音声layer 0内部正規化の固定参照照合](m4-layer-zero-norm-observation-validation.md)
- [固定native layer 0のfeed_forward1内部参照](m4-native-feed-forward-reference-validation.md)
- [固定入力による音声layer 0内部参照](m4-native-layer-replay-validation.md)
- [M4: native SiLU保持と標準ONNX Swishの小さい比較](m4-native-silu-swish-probe-validation.md)
- [M4: 固定PCM入力境界のTDD検証](m4-pcm-input-validation.md)
- [固定native入力によるpost norm単独検証](m4-post-norm-isolation-validation.md)

</details>
