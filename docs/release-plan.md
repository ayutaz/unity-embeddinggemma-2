# 正式リリースまでの計画

確認日: 2026-10-18。詳細は[ロードマップ](roadmap.md)、対象範囲は[ゴール](goal.md)に従います。

## 現在地

- 正式Releaseは未作成。PR #86・#87・#89・#90はすべてDraft・未マージです。
- 音声の別世代候補は[全6信号の数値CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39106618371)が成功。2 profile・824,320値で元の許容差超過は0、取得した実データの独立再監査も一致しました。
- この結果は候補の実GPU・consumer・Playerの合格を意味しません。元の音声モデル2件のbyte同一復旧は未達で、#89のRequired CIは失敗しています。
- #90のPlayer検証基盤はEditor契約・元fixture読み込み79件が成功。実モデルを使うWindows Release Player検証は残っています。#90のCIは確認時点で進行中ですが、固定元モデル取得の6件と集計はすでに失敗しています。
- 以前のsource・既存bundleでのGPU動作実績は保持し、最終候補に対応する確認と区別します。

### 音声候補の保存checkpoint

[保存CI 39109531607](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39109531607)が成功しました。デバッグ情報だけを除いた配布用コピーについて、両profile・全6信号のnative出力が除去前後でbit一致しています。配布artifact `12129783259` は `2027-01-16T07:56:21Z` まで保存されます。[読み戻し記録](results/audio-candidate-distribution-ci-readback-20261018.json)は公開manifestのCRCと両モデル内のmetadataを独立確認しています。アップロードされたZIP全体のSHA確認は別の実行中の取得で行い、この記録では未完了です。

候補producerの元graph2件・raw入力6件・JSON2件の計10ファイルは、元のhashで私的な2コピーを確認しました。消失した旧世代graphの復旧とは区別します。公開branchは `5a064be`、保存実行sourceは `6bfc223`、統合候補のRuntimeは未変更です。必須CI・実GPU・consumer・Playerの合格には読み替えません。

次は取得中のZIP全体の照合、元の数値参照とraw入力も短期artifactに依存せず取得できる経路、配布用graphからのGPUモデル準備を進めます。モデル2件を保存しただけでは、全6 moduleの公開準備手順は完了しません。

## 実行順と完了条件

| 順序 | 実施内容 | 完了の判断 |
| --- | --- | --- |
| 1 | 音声候補と参照データの保存・取得経路を確保し、候補の実GPU・consumer・Player検証を未統合branchで実施。配布モデル内の絶対パスを含むデバッグ情報も監査 | 原本の参照値・許容差を維持して必要な検証が成功。配布用にデバッグ情報を除く場合は計算と重みの不変を検証し、変更後のhashを別途固定。必須CIを条件変更で迂回しない |
| 2 | 新規利用者向けのモデル準備を確立。空checkout・空cacheからtext・画像3・音声2の全6 moduleを準備 | 公開手順だけで準備でき、source・model hashが一致。欠落・破損・中断・期限切れから復帰できる |
| 3 | 日英検索、画像プレビュー、音声再生・途中停止、実マイク、キャンセル・再試行・解放を最終候補で確認。性能の不足を修正 | 権限拒否・deviceなし・非対応入力・中断復帰を含め成功。初回/warm、CPU/GPU時間、RAM/VRAMを測定範囲とともに記録 |
| 4 | WindowsのIL2CPP / Release / High stripping Playerで3入力・全必要role・横断検索を確認 | 元参照のvector・順位・score、実GPU使用、fallback/readback監査、独立cold/warm起動、解放・正常終了を確認 |
| 5 | macOS EditorとiOS / Android実機で受け入れ | 3入力の実GPU結果、権限、suspend/復帰、メモリを端末・OS・graphics APIごとに確認。未実行・skip・CPU fallbackをGPU合格に数えない |
| 6 | 最終source・モデル・依存を固定し、配布物・README・API・sample・CHANGELOG・利用条件・最低要件を整備 | 最終候補で全必須回帰が成功。履歴・公開payload・画像の目視/OCR/hashを監査。旧PR/cacheの個人情報対応も追跡 |
| 7 | version・tag・GitHub Releaseをそろえて公開し、公開tagから導入し直す | 公開物の読み戻し、新規consumer導入、sample smokeが成功 |

工程1と工程2〜4の検証は重なるため、未統合候補で先に検証を進めます。新モデルの採用条件を満たした後に取得pinと必須CIを復旧し、最新headのCI・差分・privacy確認を経て #89 → #86 → #87を統合します。#90はbaseである#89の統合後に差分を再確認し、Player検証の根拠とともに統合します。マージの順序と検証の開始順を混同しません。

## 並行準備と日程

- macOS・iOS・Androidの実機と署名・build環境を先に確保します。現在記録されているAndroid接続端末とself-hosted runnerは0、Mac・iOS環境は未確定です。実行開始時に再確認します。
- 重い取得・変換・build・回帰はGitHub Actionsを優先し、ローカルUnityの重複起動を避けます。同じモデル探索や生成を根拠なく繰り返しません。
- 各checkpointでは「合格・失敗・未実行」「実行source/model」「次の残件」を記録します。テスト数だけでリリース進捗を表しません。
- リリース日は未確定です。音声候補のGPU採用条件と必須端末の実行見通しが立った時点で、実測時間に基づき日程を更新します。端末未確保のまま完了日を約束しません。
- 公開表示名 `yousan` は維持します。元画像や元の数値証拠を加工・再生成して成功の証拠に置き換えません。
