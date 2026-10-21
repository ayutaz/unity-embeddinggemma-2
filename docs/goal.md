# プロジェクトのゴール

更新: **2026-10-21**。最初のWindows正式版 **[v0.1.0は公開済み](https://github.com/ayutaz/unity-embeddinggemma-2/releases/tag/v0.1.0)** です。
実際の受け入れ範囲は[検証状況](status.md)、今後の作業は[ロードマップ](roadmap.md)を参照してください。

## 目指すもの

Unity Sentisだけでテキスト・画像・音声をGPU上で共通の正規化768次元vectorへ変換し、モダリティをまたぐ検索を提供します。
追加のllama.cpp・ONNX Runtime・LiteRT等のネイティブ推論プラグインはRuntimeへ導入しません。
Texture / RenderTexture・PCMの前処理、encoder、共通backboneをGPUでつなぎ、中間tensorのCPU readback・計算fallbackを禁止します。

tokenize、ファイルdecode、PCM取得 / uploadはCPU側の入力境界、最終768次元配列の取得は出力境界です。
GPU非対応時のCPUへの自動切替はしません。テキストAPIの明示的CPU条件・Python参照のCPU実行を、製品のGPU受け入れと分けます。

## 維持する条件

| 項目 | 基準 |
| --- | --- |
| 最終vector | 有限768次元・単位長。固定元参照に対しFP32 cosine >= 0.999、Float16重み >= 0.99 |
| encoder・中間値 | 固定model revision・元参照・元の要素別許容差を維持。失敗へ合わせた期待値変更をしない |
| 入力と検索 | document / query / raw、対応する全固定profile、混在索引の全順位・scoreと同点処理を確認 |
| 実GPU | actual backend・計算fallback・中間readbackを監査。import、tiny graph、skip、Python CIを実モデルGPUの合格にしない |
| 導入と寿命 | 空consumerの固定tag導入、モデルのsize / hash・法的表示、キャンセル・回復・切替・解放・正常終了を確認 |
| 性能と機器 | source・モデル・仕事量・機器を固定し、測定範囲と欠測を明示。観測peakや検証構成を物理的最低容量へ読み替えない |
| 配布とprivacy | SDKコードとモデルを分離し、モデル重み・大きい生成物をGit / LFSへ入れない。送信履歴・実payload・identity・画像を監査 |

## 到達点と次の段階

Windows Editor / Playerでは、UPM・検索sample、text・image・audioの両形式、固定tag導入と正式公開まで受け入れました。
対応構成・数値・検証の限界は[正式版の確認記録](results/windows-v010-release-20261021.json)と[対応構成](windows-supported-configuration.md)へまとめています。
今後はWindows保守、必要な参照比較・性能確認、環境が揃った他platformの受け入れを進めます。
macOS / iOS / Androidは未対応で、端末・署名・ライセンス不足をWindows公開の待ち条件に戻しません。

固定入力の一致と、任意入力の意味検索品質は別の評価です。Float16重みの採用は全演算FP16化や一般的な高速化を保証しません。

## 対象外

- 動画の埋め込み、学習・fine-tuning、本格的なベクトルDB。
- ネイティブ推論プラグインによる代替実装。
- Web / WebGPUを現在の正式対応範囲へ追加すること。

開発は[AGENTS.md](../AGENTS.md)に従います。過去の固定sourceの結果はそのまま保持し、未実行の機器・sourceへ成功を拡張しません。
