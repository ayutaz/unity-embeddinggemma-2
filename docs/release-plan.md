# 正式リリースまでの計画

確認日: 2026-10-18。対象はテキスト・画像・音声のGPU推論と横断検索です。[ゴール](goal.md)の必須環境・精度条件を維持します。

## 現在地

正式tag・GitHub Releaseは各0件。PR #86・#87・#89・#90はDraft・未マージです。以下は同日の実データ・GitHub照合に基づくcheckpointであり、正式候補全体の合格ではありません。

- 新しい音声候補は、source `b6ede71` の[変換CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39120142716)で両profileの生成に成功。保存ZIP全体のCRC・hashと私的な2コピーを確認しました。
- 固定Git consumerのsource `a164cba` で、全6信号・824,320値を実GPU照合し、元の要素別許容差超過は0でした。
- source `c1e9aa8` で、実PCMから全6信号×Query / Document / Rawの18最終vectorを実GPU照合しました。全6processはexit0、skip0。元のcosine条件0.999に対する最小値は0.9999997097692651でした。破損control18件も拒否しました。
- 先行する15条件まとめ実行は既定180秒を超えて失敗しました。この失敗を保持した上で、上限を変えず1信号3条件ずつに分割して上記の合格を確認しています。
- source `39be89d` で保存bundleを生成。434ファイル・285個の固有Sentisモデルのhash、operation / coefficient対応、私的な2コピーを検査しました。モデル記述内の文字列とtokenizer JSONも監査しました。
- 保存bundleのEditor GPU照合はDocumentの全6信号で成功。元参照を独立再計算し、最小cosineは0.9999998052055729、破損control6件を拒否しました。このgateだけでは保存bundleの全roleやPlayer合格を意味しません。
- 新bundleのWindows IL2CPP / Release / High stripping Playerは、両音声profileのcontrollerが成功終了を記録しています。実build sourceは `89b1ab9`、bundle生成sourceは `39be89d`。Runtime Git tree一致により既存binaryを再使用しており、新buildとは扱いません。capture・元参照・全Worker監査を含む独立再監査は残件です。
- 既存bundleではsource `89b1ab9` の実Playerで画像3・音声2×FP32 / Float16重みの10条件が合格しています。新候補や最終統合版へそのまま付け替えません。
- #89 head `39be89d` の[通常CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39123968041)は確認時点で進行中ですが、固定音声6信号と集計の失敗が既にあります。必須CIの合格は未達です。旧native PT2の同一bytes復旧も未達であり、新候補の成功とは区別します。

## 優先順位と完了条件

| 順序 | 実施すること | 完了条件・成果物 |
| --- | --- | --- |
| 1 | 新しい音声候補の採用条件と取得pinを確定し、必須CIを復旧 | 新保存bundleのPlayer独立監査、実際にFloat16変換したbundleの対応・GPU照合、固定元参照との一致、配布モデルのprivacy確認、再現可能な取得・生成をそろえる。元参照・許容差を変更せず、最新headの必須CIが全成功 |
| 2 | 空checkout・空cacheから利用者向けの準備を再現 | 公開手順だけでtext・画像3・音声2の全6 moduleが準備でき、source / model hash一致。欠落・破損・中断・期限切れから回復 |
| 3 | 最終候補のsample操作と性能を確認 | 日英検索・横断検索、画像、音声再生 / 停止、実マイク、キャンセル・再試行・解放が成功。権限拒否・deviceなしも確認。初回 / warm、CPU / GPU時間、RAM / VRAMを実測し最低要件を決定 |
| 4 | 最終統合版のWindows Playerを受け入れ | IL2CPP / Release / High strippingで3入力・全必要role・横断検索のvector / 順位 / scoreが元参照と一致。実GPU・fallback / readback・独立cold / warm起動・解放・正常終了を確認 |
| 5 | macOS Editor・iOS / Android実機を受け入れ | 端末・OS・graphics APIを記録し、3入力の実GPU・権限・suspend / 復帰・メモリを確認。skip・未実行・CPU fallbackをGPU合格に数えない |
| 6 | 最終配布候補を固定し、公開前監査 | 必須回帰全成功。README / API / sample / CHANGELOG / 利用条件 / 最低要件を整備。履歴・commit identity・実公開payload・画像の目視 / OCR / hashを確認。旧PR / cacheのprivacy対応も追跡 |
| 7 | 正式公開と公開後の導入確認 | version・tag・GitHub Releaseを一致させ、公開物を読み戻す。公開tagを新規consumerへ導入し直し、モデル準備とsample smokeが成功 |

工程1は進行中です。工程2〜4には既存sourceでの部分的な実績がありますが、最終候補としての完了判定は残っています。工程5〜7は未完了です。

## PRの整理

- #89: privacy防止と音声候補・取得経路の復旧を完了させます。最新headの必須CI・差分・privacy監査後に統合します。
- #86: 新候補の固定取得先を反映し、空cache取得・破損回復を確認して統合します。
- #87: 採用Float16モデルの対応と性能測定を最終候補へ結び付け、必要な回帰後に統合します。
- #90: baseの#89統合後に差分を再確認し、Player監査の根拠とともに統合します。

想定順は #89 → #86 → #87 → #90。依存差分を再確認して必要なrebaseを行います。ユーザーの既存マージ指示の範囲内で、各PRの最新headが必須CIとprivacy条件を満たした時点で進めます。失敗中のCIを迂回せず、mainへ直接pushしません。新規tag / Releaseは工程6の完了後に扱います。

## 実行環境・日程・報告

- macOS / iOS / Androidの端末・署名・build環境は、工程1と並行して先に確保します。直近の環境確認は接続Android端末0、self-hosted runner0、Mac / iOS未確定です。実行開始時に再確認します。
- 重い取得・変換・build・回帰はGitHub Actionsを優先します。ローカルのUnity / GPU実行は必要な確認に絞り、1processずつ動かします。
- 次のcheckpointは「新保存bundleのPlayer独立監査・Float16確認」「必須CI全成功とPR整理」「空cache導入」の順です。各報告に合格・失敗・未実行、source / model、残件を記録します。
- リリース日は未確定です。工程1のCI復旧と工程5の実行環境の見通しが立った時点で、実測から日程を見積もります。端末が未確保のまま完了日を約束しません。
- 公開表示名 `yousan` は維持します。公開文書には相対 / 環境変数のパスだけを使い、元画像・元数値証拠は非公開で保持します。証拠画像を加工・再生成して漏えいや失敗を隠しません。
