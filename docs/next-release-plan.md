# Windows先行リリースまでの実行計画

更新: 2026-10-20。ユーザーの方針により、最初の正式リリースはWindows Editor / Playerを対象にします。macOS / iOS / Androidは今後の対応です。[ゴール](goal.md)のテキスト・画像・音声、横断検索、FP32 / 採用Float16重み、Sentis GPU、数値・運用・性能・privacy・新規導入の条件は維持します。

**工程1〜3のWindows機能検証は完了しました。** [確認記録](results/windows-first-stages-1-to-3-20261020.json)に実行sourceと適用範囲を記録しています。残る工程は4「性能・最低要件・公開前整理」と5「新tag導入・正式公開」です。正式リリース完了の報告ではありません。

## 現在地

公開mainは `3e78a27`（[#114](https://github.com/ayutaz/unity-embeddinggemma-2/pull/114)まで統合済み）。#114の必須CI全15 jobは成功し、実mainの同tree・署名・公開identity・全祖先を確認しました。同treeに対する重複main CIは中止し、成功には数えません。現時点のopen PRは0件です。以下の追加作業はtask branch上で準備中で、mainへの統合・正式公開は未完了です。

#113のhead `c980027` では実13数値archive、元824,320値とCPU encoder 120 capture / 30,822,400値を独立照合済みです。許容差超過は0件です。初回upload timeoutとその失敗記録を保持し、同sourceの限定再試行を受け入れています。

Windowsの固定候補は `e5b6bd9`。実行記録のcandidate `9967516` とリポジトリ全treeが一致することを確認して、統合済みmain上へ文書branchの履歴を整理しました。SDK実行コード・依存・設定・モデルの同一性を照合し、実Editor検証と既存Playerの結果を次の範囲で適用します。

| 対象 | 確認した結果 | 適用範囲 |
| --- | --- | --- |
| Windows Editorの実モデル | FP32 / Float16重みの各1プロセスで、全画像・音声profile / roleの基本36＋反復45ベクトル、18文書 × 4クエリの順位・scoreを照合。追加1プロセスでテキスト全roleの30ベクトルとRenderTexture全roleの12ベクトルを照合 | Unity 6000.3.21f1 / Sentis 2.6.1 / RTX 4070 Ti SUPER / D3D12。3プロセスとも正常終了。元数値条件を維持し、computation CPU fallback・中間readbackは0 |
| Windows Player | media source `f0fc2a7` の実Win64 IL2CPP / Release / High build、4 cold / warmプロセスの元結果を再照合。実行ソース71件・build全552ファイルと証拠2コピーを再確認。text source `3a02fc0` の全15ケース × GPU4条件も再照合 | 候補とのSDKコード一致を確認した既存実行の適用。新candidate SHAでの再build・再実行としては数えない。textのCPU4条件をGPU合格に数えない |
| モデル準備 | 既存の公開cold取得・中断 / 再開・破損回復の証拠を保持。現行コードで両形式の全1,640ファイルをwarm再利用し、raw descriptor保持と全hashを確認。転送・検証・再開loopのコード一致と変更箇所を照合 | 追加downloadは0。変更されたdescriptor保存処理は現行の実warm結果と、4 CI環境で各63件（うちdescriptor10件）の実XMLで確認。新しいcold公開取得とは区別 |
| 操作・回復 | 同一SDK実装のnative UI元イベントを再照合。loading cancel / retry、空入力からの回復、再生・停止、resetを確認。現Editorでpackage / sampleの560件合格、失敗・skip0（GPU名の小規模契約124件を含む） | 権限拒否・pause / disable・再開はfake device契約を実Unityで確認。OS権限設定を変更した実拒否試験ではない。native UIの既知のobserver warningを保持 |
| 明示収録 | source `8649c94` の実マイク8.4秒、GPU270 worker、有限・単位長768次元と解放の結果を保持。現候補と音声Runtime・capture実装の一致を確認 | 元sourceを付けた適用。新規収録は行わず、音声の保存・送信なし。任意音声の意味検索品質の保証にはしない |
| 他環境 | macOS / iOS / Android CIの `unity_license_missing` failure、Metal 16整数probeのみの成功を保持 | 実Sentisモデル・実機GPUは未合格。Windows公開後の別工程 |

CPUのPython参照・比較は検証用です。製品の画像 / PCM前処理、encoder、backboneはGPUで接続し、中間CPU readback・計算fallbackを禁止します。入力decode・tokenize・PCM取得と最終768次元配列の取得は入出力境界として記録します。

## 工程と完了条件

| 工程 | 状態 | 確認内容・残る条件 |
| --- | --- | --- |
| 1 #113の検証と統合 | 完了 | 必須CI・全13数値archive・元参照・公開identity・履歴・payloadを確認して統合。実mainのtree・署名・全祖先も確認済み |
| 2 Windows候補の固定とbuild | 完了 | SDK・依存・設定・両モデルdescriptorと全object hashを固定。実buildと候補の入力一致、取得・回復・切替・解放を上記source別の証拠で確認 |
| 3 Windowsの全機能を受け入れ | 完了 | Editorの実モデル・全role・RenderTexture・混在索引・横断検索と、同一SDKの既存Player結果を照合。actual backend、fallback / readback、キャンセル後の回復・解放・正常終了を確認 |
| 4 性能と公開前整理 | 未完了 | 同じhardware・source・モデル・入力・仕事量のCPU比較で不足分だけ測定。初回 / warm・各stage・RAM / VRAM・解放を分離。元数値gateを通る比較表と実測から最低要件を決め、公開物・全履歴・実payload・旧providerコピーを確認 |
| 5 Windows正式SDK公開 | 未完了 | 版・CHANGELOG・license・取得手順を確定。固定tagを空consumerへ導入し、モデル取得→全入力→検索→解放・終了を通して正式Releaseを公開。工程4と新tagの実導入を満たして判定 |

## 保持する失敗と限界

追加Editor試験の初回は検証コードのNUnit記法がcompile errorとなり、モデル実行前に停止しました。次の実行は検証用16 GiB total commit上限に達した後、D3D12 device errorで終了しました。両方を失敗として保存しています。GPU復帰後の560契約試験を確認し、受け入れた3プロセスではowned Editorのresident上限16 GiBと、GPU driver予約等を含むowned jobのcommit上限32 GiBを分けて監視しました。この監視設定はSDKの最低RAM / VRAM要件ではありません。

元の最終vector・中間値の参照と許容差は変更していません。既存の音声GPU4条件16回の[測定](results/audio-gpu-repeat-four-conditions-20261020.json)は保持します。追加の画像・音声GPU計測は10条件・実40回を完了し、各条件18区間、計180区間でメモリsampleを取得しました。[公開用集計](windows-media-memory-results.md)は元source・raw bytes・40出力の独立照合と保管2コピーを確認済みです。12 query errorと利用不可sampleは欠測のまま記録しています。区間には重なる範囲を含み、sample上の最大値は真のpeakや専有量の保証ではありません。条件別のCPU比較と初回サポート構成は下記の後続記録で整理済みです。CI統合と公開前確認は未完了です。機能検証のprocess時間を推論速度やGPU専有時間として扱いません。新tag consumer、正式版・正式Release、他hardware / platformの合格も未完了です。

## 次の進め方

### 最新のチェックポイント

2026-10-20にGitHubを再確認しました。#114はマージ済み、公開mainは `3e78a27`、open PRは0件です。#114のRequired CIは成功していますが、macOS / iOS / Androidの別workflowは失敗を保持しており、全platformの合格ではありません。追加の実装・計測記録はtask branchにあり、まだ公開mainへ統合していません。

音声CPU8回・画像CPU12回とGPU40回の計測・照合は揃いました。画像CPUは正方形16 lane、横長・縦長nativeという条件別の合格です。共通設定での全画像合格とは扱いません。RAM64 GB / VRAM16 GBクラスは検証済み構成に基づく初回サポート基準で、物理的な最低容量の証明ではありません。

CPU比較の[Windows CI経路](windows-CPU-CI.md)は実装し、関連134テスト（失敗・skip0）とactionlint 1.7.12の検査に合格しました。音声native経路と画像の条件別設定、失敗終了・生log非公開・限定artifactの契約を確認しています。GitHub上の新workflowは未実行で、実モデルCI合格としては数えません。成功済みのモデル変換や同条件の重い実行は、変更・失敗・新たな懸念がある場合に限って再実行します。

ここからは次の4段階で進めます。

1. **新CIの実行結果を確定する。** 実装・ローカル関連検査は完了です。公開停止条件の解消後、PRで音声native経路と画像の明示設定のCIを実行し、元gate・capture監査・load / release・実sourceを確認します。旧ONNX経路の未完了・失敗を成功へ読み替えません。
2. **公開前確認を終えて一つのPRへまとめる。** 性能表・対応構成・導入手順、全祖先・送信payload・公開identity・Git hooksを確認します。新規・変更画像があれば実bytesを目視とOCRで確認します。旧providerコピーの個人情報除去も確認し、必要CI成功後に許可済みのマージを行います。
3. **固定版を新規consumerへ導入する。** 版・CHANGELOG・license・tagを揃え、公開モデル取得からテキスト・画像・音声・RenderTexture、横断検索、解放・正常終了まで確認します。既存プロジェクトでの成功を新tag導入の成功として代用しません。
4. **Windows版の正式Releaseを公開する。** 前段の完了後に公開し、tag・配布物・導入手順を読み戻して照合します。macOS / iOS / AndroidはWindows公開後の別工程です。

公開停止条件は、既知の旧providerコピーの個人情報除去が未確認である点です。Support案件の読み取りは個別許可待ちで、新しい送信は行いません。ローカルでのCI実装・公開物準備は進められます。正式公開日はこの外部対応と新tag consumerの完了前には確約しません。

工程4では次の状態まで進みました。実行済みの検証と、まだ実モデルへ適用していない実装を区別します。

- テキストは既存の同一Windows PlayerのCPU / GPU各4条件を再照合し、[sourceを明記した比較](text-paired-performance.md)を記録しました。新しい推論実行ではありません。
- 画像CPUは横長・縦長が各4回とcapture auditを通過しました。正方形は4回を完了したものの、中間値gateが不合格で、全画像CPU比較の完了には数えません。
- 後続の[画像CLI固定条件](image-CPU-fixed-conditions.md)はsource `fe1cb9e` で正方形を16 lane、横長・縦長をnativeに明示して各4回実行し、元中間値・最終vector・正常終了・load / releaseを確認しました。全形状に16 laneを適用した合格ではなく、横長の同設定での4要素超過・exit1は別記録で保持しています。
- 音声CPUは既存5区間のgraph bytesを再利用できましたが、実測の20分上限と単一区間の5分診断上限で停止しました。最終数値が出ていない試行を数値不一致や合格とは扱いません。元の[準備停止](results/windows-cpu-baseline-preparation-stop-20261020.json)も保持します。
- CPU参照専用のFMA / stem実装は小規模33テストに合格しました。画像の3種類の診断は同じ3要素の中間不一致を解消しませんでした。同じ変更での再試行は止めます。
- [音声の明示的CPU経路](native-audio-CPU-results.md)は共有用CLIへ統合し、source `fd7f375` でも2profile・各4回で元の全中間gate・最終vector・正常終了を確認しました。encoderは全8回bit単位で一致し、同じ入力のGPU比較と、CPUの独立load / 係数照合 / releaseも記録しました。CI経路の統合は残っています。Unity SDKのGPU実装や元の許容差は変更していません。

残作業は次の順序と完了条件で進めます。下記の4項目は、この文書の工程4〜5を完了するための手順です。

| 優先順 | 作業 | 完了条件と次への移行 |
| --- | --- | --- |
| 1 | CPU比較のCI経路を整える | 音声native CLIと画像の明示設定をCIへ統合する。条件別の元gate・capture監査・load / releaseを維持し、旧ONNX数値検証と過去の失敗を省略した成功にはしない。ローカルの合格設定を変換・重い再試行で繰り返さない |
| 2 | 性能記録と動作要件を確定する | GPU10条件の元receipt・sample・40出力、条件別の画像CPU12 / 音声CPU8回、load / releaseは集計済み。[初回対応構成](windows-supported-configuration.md)をRAM64 GB / VRAM16 GBクラスの実測構成に固定する。これは保守的なサポート下限で、真の物理下限・他GPU・低容量構成の成功は主張しない。CIと新tag consumerの結果を最後に対応付ける |
| 3 | 公開前確認と変更の統合 | 実装・記録を一つのtask PRにまとめ、必要CI、全祖先、実payload、公開identity、必要な画像目視 / OCRを確認して、許可済みのmergeを行う。旧providerコピーの個人情報削除は別条件として確認する。新規公開物はprivacy gateを満たして進める |
| 4 | Windows正式SDKを公開する | 版・CHANGELOG・license・取得手順を確定し、固定tagを空consumerへ導入。公開モデルの取得→テキスト / 画像 / 音声・RenderTexture→横断検索→解放・正常終了を確認してReleaseを公開。公開後のtag・asset・手順を読み戻して照合する |

旧providerコピーの除去は未確認です。既存GitHub Support案件の読み取りは、このセッションの自動承認レビューで個別許可が必要として拒否されたため、許可待ちです。読み取り経路を変更して回避せず、新規メッセージも送信しません。provider対応は返答・状態変更時と公開判定前に確認します。外部対応待ちの間も、CPU比較・性能集計・公開物の準備を進めます。

- 工程4は不足する比較と公開前確認に絞ります。成功済みの同条件の変換・GPU機能試験は繰り返しません。旧providerコピーは現在の履歴の安全性と別に除去を確認します。
- 重い処理は対応するCI経路が使える範囲で優先します。Unity CIライセンス未設定の同じpreflightを繰り返さず、Personal entitlementをCIへ流用しません。
- macOS / iOS / AndroidはWindows公開後にライセンス・host・端末・署名を確保し、buildと実モデルGPUを別々に受け入れます。端末不足をWindows公開の待ち条件にはしません。
- 実装はTDD・uv・task branch / PRで進めます。公開前にはproject-privacy、versioned Git hooks、全祖先・実payloadを確認し、新規・変更画像は目視とOCRで実bytesを確認します。公開名 `yousan` は維持します。

正式公開日は未確定です。次の見積り更新点は「同条件性能比較・最低要件の確定」「公開前整理の完了」「新tag consumerの完了」です。開発作業とprovider対応の待ち時間を分けて報告します。過去のsource別記録は[検証状況](status.md)と[従来の計画](release-plan.md)に保持します。
