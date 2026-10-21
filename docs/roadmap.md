# ロードマップ

更新: **2026-10-21**。Windows版 **[v0.1.0は正式公開済み](https://github.com/ayutaz/unity-embeddinggemma-2/releases/tag/v0.1.0)** です。
新tagの空consumer導入、両重み形式の実Editor / Windows IL2CPP Player、GPU監査、公開後の読み戻しまで完了しました。
[現状](status.md)と[正式版の確認記録](results/windows-v010-release-20261021.json)を基準にし、公開前の重複計画は廃止します。

## ここからの優先順

| 順序 | 作業 | 開始条件・完了条件 |
| --- | --- | --- |
| 1 | Windows版の保守と導入文書の維持 | 不具合報告時はsource・model descriptor・入力profile・GPU / driver・再現手順を固定する。最小再現をTDDで修正し、影響する数値・cancel / recovery・解放を元基準で確認してPR化する |
| 2 | CPU参照ツールのhost差の切り分け | Intel hostの画像中間値不一致を[既存記録](windows-CPU-CI.md)から小さい固定入力へ絞る。仮説と時間 / メモリ上限を先に決め、元中間値・最終vector・load / releaseが揃った条件だけを追加受け入れする。失敗へ許容差や期待値を合わせない |
| 3 | 性能・対応機器の拡張 | 同一source・同一入力・同一仕事量での必要な比較を先に設計する。既存の機器と無料枠で実行できる場合だけ進める。stage / wall / CPU / GPU時間とRAM / VRAMの観測範囲を分け、新しい数値一致と正常終了を確認して対応表へ反映する |
| 4 | macOS / Android / iOSの受け入れ | 下記の環境条件を満たしたplatformだけを開始する。Windows公開は完了しており、他platformの準備待ちで再度保留しない |

文書・一覧整理だけを理由に、成功済みの巨大モデル取得・変換・GPU全条件を繰り返しません。
対応機器や真の最低メモリを調べる具体的な必要がない間は、[現在の検証構成](windows-supported-configuration.md)を維持します。
公開モデルの元checkpointからの再変換は、preview取得とは別の工程です。採用モデルを更新する必要が生じた場合に限り、
producer・全object hash・元参照を固定し、全6 moduleと両形式を新しいsourceで受け入れます。

## 他platformの開始条件と移行判定

| 段階 | 必要な環境と判定 |
| --- | --- |
| CIの入口 | 利用可能なUnity CIライセンス、対象Editor / build module、十分な容量と上限時間。未設定のまま同じpreflightを再実行しない |
| build / compile | macOS Editor、Android ARM64 IL2CPP / Release / High APK、iOS MetalのXcode exportと署名なしcompileを別々に記録する。buildだけで実GPU合格にしない |
| 実モデルGPU | MacのMetal能力・容量、Android / iOSの実機と接続host、必要な署名・権限を確保する。既存の小さいcompute probeやエミュレータを代用にしない |
| 対応表へ追加 | text / image / audio、全role、混在検索、元精度gate、fallback / readback、cancel / recovery・suspend / 復帰・解放・正常終了の実結果を固定sourceへ結び付ける |

現時点で追加の実機は提供されていません。端末購入、有料runner・外注・実機サービスの利用、ユーザーへの新規作業依頼をこの計画の前提にしません。
使える既存環境が揃うまでは他platformを保留し、Windowsの保守を優先します。[CI](ci.md)と[端末検証](device-validation.md)に既存経路・失敗の範囲があります。

## 各変更と次のリリースの共通条件

1. task branchを作り、挙動変更はTDDとuvで実装する。必要な範囲のテストとworkflow lintを実行し、PRのRequired CIを確認する。
2. SDK実行コード・モデル・依存・入力・許容差への影響を明示する。文書だけの変更で、旧実測sourceを新sourceの実行へ書き換えない。
3. 実行は所有process・時間・resident / commitの上限を固定する。観測timeoutで別実行を重複起動せず、元logと失敗を保持する。
4. 公開前にproject-privacy、versioned Git hooks、全送信祖先、公開identity、実payloadを確認する。新規・変更画像は実bytesを目視・OCRする。`main`へ直接pushせず、mergeは依頼された場合に行う。
5. 次版は必要な変更がまとまった時点で版・CHANGELOG・固定tagを準備する。影響範囲の実GPUと空consumer、Release公開後の読み戻しを完了してから公開完了とする。現時点で次の版番号・公開日は確約しない。

旧provider側コピーの除去は別件として延期しています。除去完了とは扱わず、新規公開物のprivacy gateも緩めません。
履歴・固定sourceの検証記録は[文書一覧](README.md)、プロジェクト全体の不変の条件は[ゴール](goal.md)を参照してください。
