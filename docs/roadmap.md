# ロードマップ

更新: **2026-10-21**。Windows版 **[v0.1.0は正式公開済み](https://github.com/ayutaz/unity-embeddinggemma-2/releases/tag/v0.1.0)** です。
公開tagからの新規導入、FP32 / Float16の実Editor・Windows IL2CPP Player、GPU監査と公開後の読み戻しは完了しています。
[検証状況](status.md)と[正式版の記録](results/windows-v010-release-20261021.json)を基準に、今後はWindows版の保守と利用時の安定性を優先します。

## 現在地と直近の順序

2026-10-21時点の追加作業は、次の状態です。正式版の受け入れ結果と、公開後の追加検証を区別します。

| 作業 | 現在の状態 | 次の完了条件 |
| --- | --- | --- |
| P0: CIの振り分け | [PR #118](https://github.com/ayutaz/unity-embeddinggemma-2/pull/118)を統合済み。元headの完全なCIは16件成功、文書jobは設計どおり対象外。統合後の[main CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39572610630)も成功 | 文書だけの軽い経路と、コード・依存・モデル・privacy・未知の変更に対する完全な回帰を維持する |
| P1: Windowsの反復利用 | image-square / Float16の実Player 15 vectorと、実sampleの登録→検索→中断→再試行→切替→解放を独立照合済み。マイク契約10件と短時間の実device制御も成功 | 音声GPUの追加反復を実行できていないため、P1全体は未完了 |
| P2: Intel CPU参照差分 | 初回の限定調査を完了。入力・依存・参照・演算設定が同じでもIntel / AMDのencoder出力が分岐する。内部の最初の演算は未特定 | host限定を維持して保留。内部layer captureを比較できる具体的な機会ができた場合に再開する |

P0はmain `ffe959c223742f88ea1e8538f883af05d2d9a2d8`への統合とCI確認を完了しました。
計画PR #117の内容と#119の文書補足はPR #118に含まれています。公開headの内容を確認し、#117と#119は個別にmergeせず閉じ、#118の1本に整理しました。
過去の記録にある「最終CI待ち」は当時のcheckpointとして保持します。上記が現在の統合・CI状況です。

P1の[追加検証記録](results/post-release-windows-stability-20261021.json)では、元の画像参照とのcosine・検索順位とscore、実Worker・解放・正常終了を照合しています。GPUメモリは観測値と欠測を記録し、保証されたpeakとは扱いません。sampleの手動GUI操作や、submitted GPU kernel内での中断を証明した結果ではありません。
合格済みの画像条件を理由なく再実行しません。
音声は対象条件の既存VRAM観測値と安全余裕を満たす場合に順番に実行します。現在の空き容量ではその条件を満たしていないため、重い音声モデルは起動せず、契約テストなどの独立した作業を進めます。
音声sampleの[入力検査](results/audio-frontend-check-20261021.md)は実Editorで合格しました。元のデコーダーと波形・マスクが一致することの確認であり、音声GPU推論の追加合格ではありません。
CPUへの切替で代用せず、未実行範囲を残します。実マイクの物理切断・復帰、実際のOS focus遷移、他GPU / 他OSも、実施するまで未検証です。

P2の[限定調査記録](results/post-release-Intel-CPU-investigation-20261021.json)は参照ツールの再現性を対象とし、製品のSentis GPU推論とは別です。
元artifactのdigest / CRCと8回のcaptureを再照合し、小さい入力で演算順序の感度を確認しました。原因となる内部演算は未特定のため、元のgolden・許容差とhost限定を維持して初回調査を終了します。
利用者に影響する修正が見つかった場合だけ、影響範囲の受け入れと次のpatch版を計画します。

## 次に進める順序

| 優先 | 作業 | 完了条件 |
| --- | --- | --- |
| 1 | 依存更新PRを1件ずつ確認 | certifi #120とPillow #121の差分・lock・互換性・公開identity・全送信祖先・payloadを確認。各最新headの必要なCIを取得し、失敗・取消・未実行を分け、merge可能か判断する |
| 2 | 音声GPUの追加安定性検証 | 必要な空きVRAMがある場合だけ固定audio-839 / Float16の実Playerと実sampleを順番に実行。元数値基準・実Worker・検索順位とscore・中断後の再試行・切替・解放・正常終了を照合する |
| 3 | 結果と利用条件の整理 | source・モデル・入力・参照hashと結果、資源観測・欠測・未実行を記録。ロードマップと対応表に矛盾がなく、追加検証の完了範囲が明確な状態にする |
| 4 | 次のpatch版が必要か判断 | 利用者に影響する修正がある場合に、影響範囲の回帰・実GPU・新規consumer確認を揃え、版・CHANGELOG・固定tag・公開後の読み戻しを準備する |
| 保留 | Intel CPU参照差分の追加調査 | 初回の限定調査は完了。新しい内部layer captureと小さい再現例を比較できる具体的な機会ができた場合だけ再開する |
| 条件付き | 性能改善・他GPU・最低メモリの検証 | 具体的な問題や利用可能な既存機器がある場合に着手。同条件の数値一致・正常終了と、測定範囲を明示した結果を揃えて対応表へ反映する |
| 保留 | macOS / Android / iOS対応 | CIライセンス・build環境・実モデルGPUを確認できる機器が揃ったplatformだけ着手する |

不具合報告で再現する問題が出た場合は、source・モデルdescriptor・入力profile・GPU / driverと再現手順を固定し、その修正を優先します。
まず1を進めます。依存PRの結果待ちや2のVRAM不足の間も、差分確認と3の事実整理は進められます。音声GPUの成功を記録する部分は2の実行後に行います。
今回の計画作成は依存更新のmergeや新版Releaseの公開を実施した記録ではありません。

## 依存更新PRの確認

[certifi #120](https://github.com/ayutaz/unity-embeddinggemma-2/pull/120)と[Pillow #121](https://github.com/ayutaz/unity-embeddinggemma-2/pull/121)は別々に判断します。
両PRのPython・model-referenceは成功しましたが、package privacyとRequired CIは失敗しています。原因と個別差分の照合は[保守の中間確認](results/windows-maintenance-review-20261021.md)に記録しました。
privacy規則を変更せず、依存差分だけを公開identityで[PR #123](https://github.com/ayutaz/unity-embeddinggemma-2/pull/123)へまとめました。新sourceの完全なCIはこのcheckpointでは実行中です。
Pillowはmajor更新のため、画像の読み込み・生成・既存fixtureとの互換性を特に確認します。
既存runのsourceと結果を読み戻し、具体的な失敗を修正する場合はTDDで対応します。同じ重いCIの再実行を増やさず、新しいcommitや修正が必要な場合だけ、そのheadの回帰を追加します。

## P0: CIの待ち時間を減らす

直近の[PR #116の必須CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39517101855)は、作成から完了まで待機込みで約77分でした。
PR #118で以下の振り分けを実装し、文書・完全な回帰の実PR確認と統合を完了しました。以下は維持する設計と受け入れ条件です。

1. 差分を分類する小さい処理を作る。まず、文書だけ・Runtime・Editor / sample・参照 / モデル・依存・CI / privacy規則・複数領域・未知のpathのテストをRedにする。
2. 文書だけの経路は、差分・リンク・公開identity・全送信祖先・実payloadのprivacy検査を必須にする。変更された画像は既存の目視 / OCRとhash照合を維持する。
3. Runtime、Editor / sample、数値テスト、モデル生成・参照、依存固定、workflow、privacy規則の変更と、判定不能な差分は完全な既存回帰へ送る。文書以外を変更して軽い経路へ入れない。
4. 集約checkの名前は`Required CI`を維持する。必須の検査の欠落・failure・cancel・予期しないskip・分類の不整合は失敗にする。明示的に対象外と判定したモデル検証は「未実行」と記録し、GPU成功に数えない。
5. 文書だけの変更と、コード変更を含むPRの実runで振り分けを確認する。後者は既存の元参照・元許容差の完全な回帰を通す。branch protection、artifact guard、画像・モデルの公開条件を維持する。

文書だけの経路は、runner待ちを除く実行時間10分以内を目標にします。PR #119の1回の実測では作成から完了まで39秒でした。他の差分や混雑時の所要時間を保証する値ではありません。
PRとmainで同じ内容を重複検証する扱いも、source・全treeの照合と必要checkを保つ条件で検討します。現行runを一律に停止する運用にはしません。

## P1: Windowsの安定性を範囲を決めて確認する

小さい契約テスト・エラー経路はCIへ置き、実GPU確認は既存のWindows構成で必要な条件から順に実行します。
初回の連続実行は**1 profile・1重み形式・30分以内**に限定し、複数のUnity / Playerを同時に起動しません。
既存のメモリguardと処理の所有関係を維持し、数値不一致・解放失敗・上限超過で停止して原本を保持します。

音声audio-839 / Float16の入口条件は、既存観測値と安全余裕に基づく**空き専用VRAM 11,815 MiB以上**です。
計画更新時の一度の観測は7,691 MiBで、入口条件を満たしていません。空き容量は実行直前に再確認します。
安全余裕を下げたりCPUへ切り替えたりして合格扱いにしません。他の作業を含むTerminalの終了は許可がある場合だけ行い、未回答を許可と見なしません。
資源が確保できるまでは音声GPU検証を保留します。監査済みモデルと実行物を再利用し、変更のない巨大graphは再変換しません。

- 画像・音声の登録とテキスト検索、キャンセル直後の再試行、モデル切替・解放・終了を確認する。
- 反復回数、入力・参照hash、元数値gate、実Worker、RAM / VRAMの観測値と欠測を記録する。観測最大値を保証されたpeakへ読み替えない。
- マイク停止・位置後退・フォーカス喪失・録音破棄・再録音の制御をまず契約テストで確認する。既存deviceで可能な実経路を別に確認し、物理切断・復帰ができない間は未検証のまま残す。
- 問題が再現した場合だけTDDで修正し、影響するprofile・重み形式と実GPU結果を追加する。合格済みの巨大モデル全条件を理由なく繰り返さない。

この段階の終了条件は、対象条件の数値・回復・解放・正常終了が揃い、未実行範囲も記録できることです。
他GPU・他OSや、任意入力の検索品質まで確認したとは扱いません。

## P2と、環境が必要な作業

[Windows CPU CIの既存記録](windows-CPU-CI.md)と初回の限定調査を保持します。追加調査を再開できる場合は、Intel hostで最初に不一致となる内部演算を1条件に絞ります。
失敗後に設定を自動変更せず、仮説を先に記録して小さい対照を作ります。新しい根拠が得られた仮説だけを既存のCIで確認します。
同じ設定の再実行で成功を探さず、golden・許容差を変更しません。製品のSentis GPU推論と参照ツールのCPU実行を区別します。

他GPU・低容量構成は、利用できる既存機器と具体的な検証目的がある場合だけ開始します。
それまでは[現在の対応構成](windows-supported-configuration.md)を維持します。元checkpointからの全モデル再変換は、採用モデル更新が必要になった場合だけ別工程として行います。

macOS / Android / iOSは、利用可能なUnity CIライセンス・対象Editor / build module・容量と時間上限を入口条件にします。
build / compile、実モデルGPU、端末の権限・suspend / 復帰・解放・正常終了を順に記録し、実GPUの元数値基準まで通った場合だけ対応表へ追加します。
実機の代わりにエミュレータや小さいcompute probeを成功へ数えません。環境不足の同じpreflightを繰り返しません。
端末購入、有料runner・外注・実機サービス、ユーザーへの新しい作業依頼は前提にしません。

## 進め方と次のリリース

P0は完了済みです。P1の各実GPU実行は30分以内とし、開始・結果・停止理由を区切って報告します。P2は最大2時間の初回調査を終了しています。
依存更新のCI待ちと音声実行のVRAM確保時間が未確定のため、全体の完了日はまだ確約しません。
現在の結果と残りの作業をこの文書に集約し、同じ内容の別計画書を増やしません。

挙動変更はTDD、Pythonはuv、重い参照生成はCIを使います。`main`へ直接pushせず、各PRのmergeは依頼された場合に行います。
公開前には[project-privacy](../.agents/skills/project-privacy/SKILL.md)、versioned Git hooks、全送信祖先・公開identity・実payloadと、画像の実bytesを確認します。
過去の実行結果は元sourceに結び付け、新sourceの再実行へ書き換えません。

利用者に影響する修正がまとまった段階で次のpatch版を判断します。版・CHANGELOG・影響範囲の実GPU・新規consumer・固定tag・公開後の読み戻しを揃えます。
現時点の[暫定判断](results/windows-maintenance-review-20261021.md)は、SDK Runtimeとpackage版に変更がなく、toolsの依存更新と文書だけで新しいSDK Releaseは不要です。音声GPUの追加検証は引き続き必要です。
CIや文書だけの改善でSDKの新しい版を必須にせず、次の版番号・公開日はまだ固定しません。
旧provider側PR / cacheコピーの除去はご指定どおり保留し、新規公開物のprivacy検査は継続します。
