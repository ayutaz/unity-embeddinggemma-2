# 正式リリースまでの計画

確認日: 2026-10-18。対象はテキスト・画像・音声のGPU推論と横断検索です。[ゴール](goal.md)の必須環境・精度条件を維持します。

直近の到達点は **#89の最新Required CI全成功** です。まだDraft・未マージで、正式リリースも未実施です。次は差分と公開履歴を確認して#89を統合し、残る3 PRの内容を1つの候補へ整理します。Windowsでの過去の成功だけで、最終候補や他OSの受け入れを完了にはしません。

## 現在地

正式tag・GitHub Releaseは各0件。PR #86・#87・#89・#90はDraft・未マージです。以下は同日の実データ・GitHub照合に基づくcheckpointであり、正式候補全体の合格ではありません。

- 新しい音声候補は、source `b6ede71` の[変換CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39120142716)で両profileの生成に成功。保存ZIP全体のCRC・hashと私的な2コピーを確認しました。
- 固定Git consumerのsource `a164cba` で、全6信号・824,320値を実GPU照合し、元の要素別許容差超過は0でした。
- source `c1e9aa8` で、実PCMから全6信号×Query / Document / Rawの18最終vectorを実GPU照合しました。全6processはexit0、skip0。元のcosine条件0.999に対する最小値は0.9999997097692651でした。破損control18件も拒否しました。
- 先行する15条件まとめ実行は既定180秒を超えて失敗しました。この失敗を保持した上で、上限を変えず1信号3条件ずつに分割して上記の合格を確認しています。
- source `39be89d` で保存bundleを生成。434ファイル・285個の固有Sentisモデルのhash、operation / coefficient対応、私的な2コピーを検査しました。モデル記述内の文字列とtokenizer JSONも監査しました。
- 保存bundleのEditor GPU照合はDocumentの全6信号で成功。元参照を独立再計算し、最小cosineは0.9999998052055729、破損control6件を拒否しました。このgateだけでは保存bundleの全roleやPlayer合格を意味しません。
- 新FP32 bundleのWindows IL2CPP / Release / High stripping Playerは、両音声profileで実行と独立再監査が成功。通常18件・in-process warm18件を元参照から再計算し、前後の全270 Worker監査、キャンセル復帰、解放・exit0と24件の破損control拒否を確認しました。実build sourceは `89b1ab9`、bundle生成sourceは `39be89d`。Runtime Git tree一致により既存binaryを再使用しており、新buildとは扱いません。
- source `c18b4e5` の新規固定Git consumerで、新候補285モデルのFloat16重み派生を生成しました。全434ファイルとFP32係数の不変、module対応、文字列privacy、私的な2コピーを独立確認。保存サイズは3,657,076,501から2,573,229,069 bytesです。この派生bundleも、両profileの実Release Playerと独立再監査で通常18件・warm18件、24破損control拒否、全GPU監査・exit0が成功しました。全演算FP16化・速度改善・独立warm起動の合格には読み替えません。
- 既存bundleではsource `89b1ab9` の実Playerで画像3・音声2×FP32 / Float16重みの10条件が合格しています。新候補や最終統合版へそのまま付け替えません。
- #89旧head `39be89d` の[通常CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39123968041)には、固定音声6信号と集計の失敗があります。最新headの合格と区別して保持します。旧native PT2の同一bytes復旧も未達であり、新候補の成功を原本の復旧とは扱いません。

7種類の証拠をcanonical JSON SHA256とSDK Runtime treeへ固定し、採用候補を使う必須数値回帰の取得経路を `58f48ac` で実装しました。27件の未実装失敗とworkflowの1件の失敗から、関連95件・actionlint成功を確認。元の全6信号・5区間・824,320値の実実行条件は維持しています。[初回選択CI 39126214004](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39126214004)は音声6信号・集計が成功し、全7 archiveの独立読み戻しでも824,320値の許容差超過0でした。ただし旧取得経路を前提とするテスト1件が各Python環境で失敗し、CI全体は不合格です。1件のローカルRedから関連111件Greenへ修正後、source `392ced7816ff5c724ccd902b0dcbf488b79cee18` の[最新CI 39127876689](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39127876689)は全15 job成功しました。Pythonの4環境、package、actionlint、通常reference、音声6信号と集計、Required CIを含みます。#89はDraft・MERGEABLE・CLEANですが未マージです。これは最新CIの成功であり、新しいGPU実行や工程1全体の完了ではありません。

全6 moduleの[実配布候補の組み立て](results/media-distribution-composition-567befb-20261018.json)はFP32 / Float16とも成功しました。各820 object・553モデルの元bytes・module対応・個人情報metadataを独立照合し、合計1,640ファイルの[実公開payload checker](results/model-publication-payload-guard-392ced7-20261018.json)も成功しました。flat HTTPS取得・回復とcompositionの関連77件が合格しています。モデルは未公開で、公開元からの空cache取得・取得後の最終consumer合格は未達です。

新配布候補の[サンプル検証入口](candidate-media-sample-validation.md)も準備しました。旧bundle用のpinと数値gateを維持し、新しいFP32 / Float16 descriptorと画像3・音声2 profileを別に検査します。未実装10件・cohort不足7件のRedから、旧policyを含む34件Green・skip0・Unity exit0と新fixtureのcompileを確認しました。これはlocal-file / no-graphicsの小規模検査で、実モデルのサンプル・PlayMode実行は未完了です。

## 優先順位と完了条件

| 順序 | 実施すること | 完了条件・成果物 |
| --- | --- | --- |
| 1 | 新しい音声候補の採用条件と取得pinを確定し、必須CIを復旧 | 新保存FP32 / Float16 bundleの対応・GPU Player独立監査まで成功。候補選択・元参照維持・必須取得経路の修正を最新CIで受け入れる。元参照・許容差を変更せず、最新headの必須CIが全成功 |
| 2 | 空checkout・空cacheから利用者向けの準備を再現 | 公開手順だけでtext・画像3・音声2の全6 moduleが準備でき、source / model hash一致。欠落・破損・中断・期限切れから回復 |
| 3 | 最終候補のsample操作と性能を確認 | 日英検索・横断検索、画像、音声再生 / 停止、実マイク、キャンセル・再試行・解放が成功。権限拒否・deviceなしも確認。初回 / warm、CPU / GPU時間、RAM / VRAMを実測し最低要件を決定 |
| 4 | 最終統合版のWindows Playerを受け入れ | IL2CPP / Release / High strippingで3入力・全必要role・横断検索のvector / 順位 / scoreが元参照と一致。実GPU・fallback / readback・独立cold / warm起動・解放・正常終了を確認 |
| 5 | macOS Editor・iOS / Android実機を受け入れ | 端末・OS・graphics APIを記録し、3入力の実GPU・権限・suspend / 復帰・メモリを確認。skip・未実行・CPU fallbackをGPU合格に数えない |
| 6 | 最終配布候補を固定し、公開前監査 | 必須回帰全成功。README / API / sample / CHANGELOG / 利用条件 / 最低要件を整備。履歴・commit identity・実公開payload・画像の目視 / OCR / hashを確認。旧PR / cacheのprivacy対応も追跡 |
| 7 | 正式公開と公開後の導入確認 | version・tag・GitHub Releaseを一致させ、公開物を読み戻す。公開tagを新規consumerへ導入し直し、モデル準備とsample smokeが成功 |

工程1は最新CIが成功し、差分・公開履歴の監査と統合を残しています。工程2〜4には既存sourceでの部分的な実績がありますが、最終候補としての完了判定は残っています。工程5〜7は未完了です。

工程2のモデル配布は正式SDK Releaseと分け、ライセンス・変更点・全ファイルhashを添えたモデル配布用previewを用意して、公開URLから実際に取得し直します。モデルはGit / LFSへ含めません。期限付きActions artifactだけに依存する数値回帰の取得経路も、同一bytesと来歴を検査して継続可能な保存先へ移します。原本が失われたものを再生成品で原本扱いしません。

## PRの整理

- #89: 最新CI成功を確認済み。差分・privacy監査後に先に統合します。
- #86: 固定取得先・空cache取得・破損回復の変更を統合候補へ取り込みます。
- #87: 採用Float16モデル・sample gate・性能測定の変更を同じ統合候補へ取り込みます。
- #90: #89をbaseとする差分とPlayer監査の根拠を確認して、同じ統合候補へ取り込みます。

まず #89 の最新Required CI・差分・privacyを確認して統合します。その後、#86・#87・#90の必要な変更を1つの統合候補へまとめ、mainを取り込んだ最新headで全必須CIと実モデルの必要な検証を行います。統合後に元の3 PRを「内容を統合済み」として整理し、個別PRをmergeした事実とは区別して記録します。ユーザーの既存マージ指示の範囲内で進めます。失敗中のCIを迂回せず、mainへ直接pushしません。正式SDK tag / Releaseは工程6の完了後に扱います。

## 実行環境・日程・報告

- macOS / iOS / Androidの端末・署名・build環境は、工程1と並行して先に確保します。直近の環境確認は接続Android端末0、self-hosted runner0、Mac / iOS未確定です。実行開始時に再確認します。
- 重い取得・変換・build・回帰はGitHub Actionsを優先します。ローカルのUnity / GPU実行は必要な確認に絞り、1processずつ動かします。
- 次のcheckpointは「最新headの必須CI全成功とPR整理」「全6 moduleの公開配布・空cache導入」「最終統合版の操作・計測」の順です。各報告に合格・失敗・未実行、source / model、残件を記録します。
- リリース日は未確定です。最新CI復旧は成功しました。次に統合候補の新規導入を1回通し、工程5の端末・署名・実行担当を確定した時点で、測定・修正・環境別検証の日程を見積もります。端末が未確保のまま完了日を約束しません。
- 次の報告は「#89の統合結果と残るPRの整理」「公開モデルからの空cache取得結果」「最終候補の操作・測定結果」の単位で行います。新しい不具合がない同一モデルの重い変換は繰り返さず、CI終了待ちと実作業時間を分けて示します。
- 公開表示名 `yousan` は維持します。公開文書には相対 / 環境変数のパスだけを使い、元画像・元数値証拠は非公開で保持します。証拠画像を加工・再生成して漏えいや失敗を隠しません。
