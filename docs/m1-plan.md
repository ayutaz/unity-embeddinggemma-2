# M1 詳細計画と完了監査

更新日: 2026-10-09。対象はWindows Editor / Unity 6000.3.16f1 / Sentis 2.6.1。

**M1のWindows Editor受け入れ条件はすべて合格した。** PR #1〜#3で基盤・参照配置・推論APIを統合し、[PR #4](https://github.com/ayutaz/unity-embeddinggemma-2/pull/4)で保存・Float16重み量子化・性能とメモリ測定を追加した。
実測コードは `329da925b01f7c9cb0b4d604055f0b9dc214c5c4`。実装の基準mainは `1f0e580d276831a6c7dd20bef7ec41fac5dacc72`。
PR #4の最終head・CI・統合状態はPR本文とChecksを参照する。

## 受け入れ条件と証拠

| 項目 | 条件 | 結果 |
| --- | --- | --- |
| 固定参照 | google/embeddinggemma-2、revision固定、batch 1 / length 128 / fp32 / 768次元 | CI成果物のsource SHA・固定15件・SHA-256を監査して配置済み |
| tokenizer | 全15件のtoken ID / attention mask完全一致 | 空文字互換処理を含め合格。長文切り詰め・Unicode・絵文字・空白も含む |
| import / fp32 | `.pt2` import成立、CPU / GPUCompute各15件でcosine >= 0.999、finite・768次元・単位長 | 両backend合格。GPUはRTX 4070 Ti SUPER / Direct3D12で実行、skip・CPUへの切り替えなし |
| 公開C# API | query / document / raw、タイトル、再利用・Dispose・不正入力 | 実モデル各15件・別入力後の再推論と単体契約が合格。UnityEditor参照なし |
| fp32保存 | `.sentis`保存・ロード、両backend各15件で保存前後の最大絶対差 <= 1e-5 | 最大差0、Python参照cosine >= 0.999を維持 |
| Float16重み | 別ファイル保存・ロード、両backend各15件でcosine >= 0.99 | 全件合格、保存サイズ約49.45%削減。推論速度改善は確認されなかった |
| 測定 | import / 準備 / 初回 / 定常推論、保存サイズ、メモリと環境 | 強制同期再import、warmup 2周 / 測定3周で各条件45サンプル、段階別メモリを記録 |
| TDD / CI | 意図したred、green、失敗と未実行の区別 | C#契約30件、completion 1件合格、failed / skipped / inconclusive=0。Python CIは4環境各64件・実モデル15件合格 |

[完了検証記録](m1-completion-validation.md)と[数値レポート](results/m1-complete-windows-20261009.json)に60比較、180測定値、37ファイルのLF正規化ソースhash、成果物hash、NUnit結果を保存。
配置・API導入時の記録は [実モデルAPI検証](m1-runtime-validation.md)、基盤実装の失敗履歴は [基盤検証](m1-validation.md)。
初回completionは数値処理終了後にNUnit既定180秒で失敗した。TimeoutAttributeを明示した再実行の成功だけを受け入れる。

## 固定する入力と参照条件

- モデル: `google/embeddinggemma-2`、revision `914f7f89142e33e77833254d9c9b90c3cef7303b`。
- 入力: [tools/cases/text.json](../tools/cases/text.json)の15ケース。日本語 / 英語のquery・document、タイトル、空文字、空白、改行、混在、絵文字、記号、Unicode、長文切り詰め。
- batch 1 / sequence length 128。入力はint32の `input_ids` / `attention_mask`、出力は有限な `[1,768]`、L2 norm 1±0.001。
- 本体の512→768 projection、プロンプトを含むmean pooling、L2正規化をexport内に保持する。
- queryは `task: search result | query: `、documentは `title: none | text: `（タイトルnullのみnone）、rawは前置きなし。
- Python参照はtransformers / sentence-transformers。生成コード・uv.lock・固定入力は元の基準main `60f906d` から変更なし。
- ローカル参照の生成元checkoutは `b3b0d76e1d3b6f4e44af1e9ba810d5b3cb6b051d`、成功run `37755304407`。参照・tokenizer・`.pt2`のhashは完了レポートに記録。

## 段階別の判定

| 段階 | 状態 | 維持する条件 |
| --- | --- | --- |
| P0 環境 | 完了 | Editor / Sentis / uloopの版を記録。依存変更時は再確認 |
| P1 参照仕様 | 完了 | 入力条件を変える時は参照を再生成し全件照合 |
| P2 参照生成・配置 | 完了 | 成功CI・固定revision・source・hashを監査。配置合格とSentis合格を分ける |
| P3 tokenizer | 完了 | 空文字を含む全15件完全一致を維持 |
| P4 export / import | 完了 | `.pt2`経路が成立。条件変更で失敗した場合のみ同じ重みのONNXを検討 |
| P5 inference / API | 完了 | 両backend・全ケース・再利用・リソース所有権の契約を維持 |
| P6 保存・量子化・測定 | 完了 | 保存前後・量子化版の精度と測定を別々に記録 |

## 測定の範囲と限界

測定条件はRyzen 9 5900X / RTX 4070 Ti SUPER / Direct3D12 / Windows 11。
GPU定常中央値はfp32 50.26ms、Float16重み58.20ms。CPUは481.43ms / 865.12ms。公開同期APIのtokenizationから最終ベクトルreadbackまで含む。
importは既存cache環境での強制同期再importであり、cold installとは区別する。
メモリはUnity native allocated / reserved、managed heap、graphics driverのEditor全体の段階別サンプル。モデル専有・連続観測peakではなく、重なる指標は合計しない。
ProcessカウンタはMonoで0を返し未取得。利用可否をレポートのflagで明示する。
Float16は保存する重みの精度であり、全演算のfp16化・速度改善・専有VRAM削減を主張しない。

## 次の作業

1. PR #4の最終head・CI・差分・実測ソースの対応を監査して統合する。ユーザーの「M1の最後まで」の範囲で実施。
2. M2の詳細計画: UPM構成、テキスト検索サンプル、配布手順、macOS Editor / iOS / Androidの精度・速度・メモリ検証。
3. M3で画像モデルとGPU前処理、M4で音声モデルとGPU前処理・モダリティ横断検索。

main保護のサーバー設定と、全PRで起動する必須CIの整合は開発基盤の別作業として残る。
クラウドUnity workflowはLinux CPUの任意手動補助検証で未実行。Secrets未登録はローカルM1の妨げではなく、Linuxや他環境を合格扱いにしない。

## 開発・再実行規則

- TDD: テスト → 意図した失敗の実行確認 → 最小実装 → green。
- Pythonはtools/のuvのみ。モデル取得・参照生成・変換・重いPython実行はActionsを優先し、実GPUは既存ローカルEditor / uloopを利用。
- mainへの直接pushは禁止。作業ブランチ・PRを使い、今回以外のmergeは依頼があるまで行わない。
- モデル・大きいログはGit管理外。Git LFSを使わない。再生成手順と小さい数値要約を残す。
- 参照artifactの保持は3日。失効・モデル生成コード・依存・入力条件の変更時はCI再生成し、新しいsource SHA / hashを監査する。
- 再実行: [uloopハーネス](automation.md)、[保存・推論API](runtime-api.md)、[完了作業計画](m1-completion-plan.md)。
- 未実行、GPU skip、tiny model、静的検査だけで実モデルの合格を代替しない。
- このM1合格はWindows Editor限定。macOS / iOS / Android、UPM公開、画像・音声は未完了。
