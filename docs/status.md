# 現在の対応範囲と検証状況

更新: **2026-10-21**。Windows版 **[v0.1.0を正式公開済み](https://github.com/ayutaz/unity-embeddinggemma-2/releases/tag/v0.1.0)** です。
公開tagのsourceは `91e3419571432dfa54e7bbab045ef63a7cef4a81`、package versionは `0.1.0` です。
以下はその版の実行結果です。文書整理後のsourceでモデルを再実行したという意味ではありません。

[導入](../README.md#導入) / [対応構成](windows-supported-configuration.md) / [今後の計画](roadmap.md)

## 確認済みの範囲

| 項目 | 実行結果 |
| --- | --- |
| 固定tagの導入 | Libraryをコピーしない空consumerで公開Git tagを解決し、実PackageCache・依存・sample bytesを固定sourceと照合 |
| テキスト | FP32 / Float16、document / query / rawの元15ケース。EditorとWindows Playerで確認 |
| 画像・音声 | 両形式の画像3形状・音声2profile、全role。前処理→encoder→backbone→正規化768次元を実GPUで確認 |
| RenderTexture | Editorはsquareの全role、Playerは全3形状・全roleを両形式で確認。Playerは各形式51実呼び出し |
| 横断検索 | テキスト6・画像6・音声6の混在索引、4 query、全72順位 / scoreを元参照と照合 |
| Windows Player | Win64 / IL2CPP / Release / High stripping。mediaの独立cold / warm各2形式、RenderTexture各形式、text 2起動の計8実processが正常終了 |
| GPUと寿命 | actual Worker backend、計算CPU fallback・中間readbackなし、キャンセル後の再実行、解放・正常終了を監査 |
| モデル | 公開preview由来の全6 module、各形式820 objectのsize / SHA256、LICENSE・NOTICE・MODIFICATIONSを確認 |
| 公開 | 正式版・latest・tagのSHA・package版・ライセンス・実Release本文を公開後に読み戻して一致確認 |

固定元参照のFP32 cosine >= 0.999、Float16重み >= 0.99、有限・単位長、検索scoreの絶対誤差 <= 0.001を維持しています。
元encoderの要素別許容差も変更していません。根拠と数値は[正式版の確認記録](results/windows-v010-release-20261021.json)、
検証前の工程1〜3は[当時の記録](results/windows-first-stages-1-to-3-20261020.json)を参照してください。

## 対応構成と制約

Windows 11 x64、Unity 6000.3.21f1、Sentis 2.6.1、D3D12、RTX 4070 Ti SUPER、
Ryzen 9 5900X、RAM64 GB / VRAM16 GBクラスが検証済み構成です。
より小さいメモリ構成が動かないと証明した物理的最低容量ではありません。他GPU・他Unity版の成功を含めません。
macOS / iOS / Androidは未対応です。既存の小さいMetal probe・APK build・単体テストを実モデルGPUの合格にしません。

画像・音声は[固定入力profile](../Packages/com.ayutaz.embeddinggemma/Samples~/MultimodalSearch/README.md)を使います。
tokenize・decode・PCM取得はCPUの入出力境界で、最終vectorはCPU配列へ返します。
テキストAPIの明示的CPU条件は診断と分け、GPU非対応時に自動fallbackしません。
全演算FP16化、任意入力の検索品質、普遍的な高速化倍率、長期運用、他機器の成功は今回の結果から保証しません。

## 結果の読み方と保持している失敗

- PlayerのRenderTextureは非公開の追加検証harnessでSDKへ実入力を渡しました。harnessの付随入力metadata保存不足と初回監査失敗を保持し、全28入力原本・固定参照・ビルドした入力検証コードを別監査で照合しました。実行原本・数値基準は変更していません。音声は同版の通常cold Playerの実出力を再利用し、RT process内の音声再推論とは数えません。
- text専用Playerの全15ケースは実Runtime内で数値判定し、保存queryの順位 / scoreを独立照合しました。保存していない全raw vectorの独立再計算や、そのmatrixでの直接Worker割当監査は主張しません。同じtextモデルのWorkerは混在検索で別途監査しています。
- 新規公開モデルのcold取得sourceと、最終clientのwarm再検査sourceを分けています。HTTP 500による中断・再開も保持し、新tagで一度にcold network取得した成功とは表示しません。
- Required CIは全15 job成功、実13 archiveを独立照合しました。別workflowの[Windows CPU参照](windows-CPU-CI.md)はsource・hostを限定します。Intel hostでの画像不一致は未解決で、許容差を緩めていません。これはSDKのCPU fallbackではありません。
- 他platformのCIライセンスfailure・実機未実行、過去の数値不一致・timeout・欠測は各sourceの記録として残します。[性能とメモリ](windows-media-memory-results.md)は観測範囲に限定し、欠測を0へ置換しません。
- 旧provider側PR / cacheコピーの除去は延期事項です。現在の送信履歴・実payload・公開identity・画像bytesのprivacy検査とGit hooksは維持します。

## 記録と次の作業

[文書一覧](README.md)から、利用手順、性能、CI、固定sourceの検証記録を参照できます。
過去の成功・失敗JSONを現在のsourceの再実行へ読み替えません。今後の実行順と開始条件は[ロードマップ](roadmap.md)へ集約します。
