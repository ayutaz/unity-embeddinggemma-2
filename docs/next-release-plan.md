# Windows先行リリースまでの実行計画

更新: 2026-10-20。ユーザーの方針により、最初の正式リリースはWindows Editor / Playerを対象にします。macOS / iOS / Androidは今後の対応です。[ゴール](goal.md)のテキスト・画像・音声、横断検索、FP32 / 採用Float16重み、Sentis GPU、数値・運用・性能・privacy・新規導入の条件は維持します。

同日のユーザー指示により、force push後もGitHub側に残る旧コミットのPR / cacheコピーの除去は後回しにし、今回のリリース待ち条件から外します。削除完了とは扱いません。Support案件の読み取り・返答待ちや旧コピーの反復調査で工程を止めません。現在の送信履歴・新規payload・公開identity・画像bytesのprivacy検査は維持し、新しい漏洩が見つかった場合は公開を止めて修正します。

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
| 4 性能と公開前整理 | 未完了 | 条件別CPU / GPU比較・初回サポート構成は整理済み。残る新CIの実結果、公開物・送信全履歴・実payloadを確認する。GitHub側の旧providerコピー除去はユーザー指示で後回し |
| 5 Windows正式SDK公開 | 未完了 | 版・CHANGELOG・license・取得手順を確定。固定tagを空consumerへ導入し、モデル取得→全入力→検索→解放・終了を通して正式Releaseを公開。工程4と新tagの実導入を満たして判定 |

## 保持する失敗と限界

追加Editor試験の初回は検証コードのNUnit記法がcompile errorとなり、モデル実行前に停止しました。次の実行は検証用16 GiB total commit上限に達した後、D3D12 device errorで終了しました。両方を失敗として保存しています。GPU復帰後の560契約試験を確認し、受け入れた3プロセスではowned Editorのresident上限16 GiBと、GPU driver予約等を含むowned jobのcommit上限32 GiBを分けて監視しました。この監視設定はSDKの最低RAM / VRAM要件ではありません。

元の最終vector・中間値の参照と許容差は変更していません。既存の音声GPU4条件16回の[測定](results/audio-gpu-repeat-four-conditions-20261020.json)は保持します。追加の画像・音声GPU計測は10条件・実40回を完了し、各条件18区間、計180区間でメモリsampleを取得しました。[公開用集計](windows-media-memory-results.md)は元source・raw bytes・40出力の独立照合と保管2コピーを確認済みです。12 query errorと利用不可sampleは欠測のまま記録しています。区間には重なる範囲を含み、sample上の最大値は真のpeakや専有量の保証ではありません。条件別のCPU比較と初回サポート構成は下記の後続記録で整理済みです。CI統合と公開前確認は未完了です。機能検証のprocess時間を推論速度やGPU専有時間として扱いません。新tag consumer、正式版・正式Release、他hardware / platformの合格も未完了です。

## 次の進め方

### 最新のチェックポイント

2026-10-21の追加CIで、画像CPU参照のIntel host依存の中間値不一致を確認しました。AVX2固定とMKL COMPATIBLEの2試行は採用せず、元のworkflowへ戻します。[33ファイルのsource照合](results/windows-CPU-CI-source-qualification-20261021.json)で、数値実装・依存・workflowが、実音声8 / 画像12回を独立監査したbaselineと一致することを確認しました。baselineの実行sourceとhostの範囲を維持し、Intel CPUの新たな合格や全CPU共通再現性は主張しません。未完了・失敗したCIも保持します。残る正式公開の判定は、最終sourceのRequired CI、公開tagを空consumerへ導入したWindows GPU Editor / IL2CPP Player、Releaseの公開・読み戻しです。対応する入力・role・参照・許容差・検証済みGPU構成は変更しません。

PR #115のhead `53c58cd`では、Windows音声2・画像3 jobが成功し、全20 captureを元のgateで独立照合しました。Required CIは未完了です。Windows Python 3.14は2,740成功・9 skipの後、20分のjob上限で中断しました。失敗したassertionは観測していませんが、完走した合格とは数えません。このmatrix要素だけ上限を30分へ変更し、他要素は20分を維持します。テスト選択・数値基準・失敗判定・元の音声6条件は変更しません。actionlintとYAMLの意味比較で、変更が上限だけであることを確認しました。修正後の新CI成功をmerge条件とします。

2026-10-20にGitHubを再確認しました。#114はマージ済み、公開mainは `3e78a27`、open PRは0件です。#114のRequired CIは成功していますが、macOS / iOS / Androidの別workflowは失敗を保持しており、全platformの合格ではありません。追加の実装・計測記録はtask branchにあり、まだ公開mainへ統合していません。

音声CPU8回・画像CPU12回とGPU40回の計測・照合は揃いました。画像CPUは正方形16 lane、横長・縦長nativeという条件別の合格です。共通設定での全画像合格とは扱いません。RAM64 GB / VRAM16 GBクラスは検証済み構成に基づく初回サポート基準で、物理的な最低容量の証明ではありません。

CPU比較の[Windows CI経路](windows-CPU-CI.md)は関連134テストとactionlintの後、PR #115のhead `7a33083` / 実行source `0104401`で音声2・画像3 jobが成功しました。全5 archiveのdigest・CRC・sourceを照合し、元中間値と最終vectorの全20回を独立に再計算しました。異なるCI hardwareの時間を同一実機GPUとの速度比較には使いません。成功済みの同条件の重い実行は必要な変更等がある場合に限って再実行します。

新規公開モデル取得ではFloat16が95 object後にHTTP 500で停止し、明示的な再実行でも同じURLが500でした。失敗を保持し、同じpathへの別queryでは元38,840 bytes / SHA一致を確認しました。5xx時のみ1回再要求する取得修正をTDDで追加し、元サイズ・hash・TLS・失敗条件を維持した実HTTPSを含む関連68テストが成功しています。実公開objectの追加probeは通常のHTTP200であり、自動再要求を観測したとは主張しません。修正後の必要CIと新tag consumerは引き続き必要です。

ここからは次の4段階で進めます。

1. **準備済みの変更を一つのPRにしてCIを実行する。** 送信全祖先・実payload・公開identity・Git hooksを再確認してtask branchをpushし、Required CIとWindows音声native / 画像条件別CIを実行します。ローカル134テスト・配布34テスト・workflow lintは完了済みです。hostedの元gate・capture監査・load / release・実sourceを確認し、旧ONNXの未実行・失敗は保持します。失敗した場合は原因と変更範囲を特定し、必要なjobだけを再確認します。
2. **CI成功後にマージし、公開候補を固定する。** すでに許可されたマージを明示的noreply authorで行い、実mainのtree・identity・必要CIを読み戻します。準備済みの0.1.0、CHANGELOG、license、性能表・対応構成を揃え、検証済みSHAへv0.1.0 tagを固定します。既存tagを移動しません。
3. **そのtagを空のUnityプロジェクトへ導入して実検証する。** Unity6000.3.21f1 / Sentis2.6.1で公開モデルの取得から、両形式のテキスト・全画像 / 音声profile・RenderTexture、横断検索、キャンセル / 回復、解放・正常終了まで確認します。同じconsumerのWindows IL2CPP Playerも確認し、GPU backend・計算CPU fallbackなし・中間readbackなしを監査します。既存sourceの成功を新tagでの実行に代用しません。
4. **Windows版の正式Releaseを公開する。** 新tagの必須結果を確認して候補表記を確定し、Release notesと導入手順を整えて公開します。tag・package version・配布物・公開手順を読み戻して照合します。モデル重みや内部検証Player・生logはSDKへ添付しません。macOS / iOS / AndroidはWindows公開後の別工程です。

旧providerコピーを理由にした公開停止は、ユーザーの明示指示で解除します。残る実行上の条件は、新CIの実結果と新tag consumerのEditor / Player検証です。次の報告点はPRとCIの実行結果とし、成功済みの同条件の変換・重い検証は変更等による必要がある場合に限って繰り返します。

版・CHANGELOG・導入文書は`0.1.0`候補として準備しました。[ローカルの配布監査](results/windows-v010-local-preparation-20261020.json)は34テストと両sampleの86ファイル一致を確認済みで、Runtimeは不変です。[最終確認項目](windows-release-checklist.md)に新tagのEditor / Player・モデル取得・全入力・検索・解放・正式Releaseの未完了条件を残しています。公開mainの版更新・tag作成・Unity実行・正式公開は今回行っていません。

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
| 1 | PR化と新CI実行 | 準備済み音声native CLI・画像の明示設定を一つのPRでCI実行する。条件別の元gate・capture監査・load / releaseを維持し、旧ONNX数値検証と過去の失敗を省略した成功にはしない |
| 2 | 性能記録と動作要件を確定する | GPU10条件の元receipt・sample・40出力、条件別の画像CPU12 / 音声CPU8回、load / releaseは集計済み。[初回対応構成](windows-supported-configuration.md)をRAM64 GB / VRAM16 GBクラスの実測構成に固定する。これは保守的なサポート下限で、真の物理下限・他GPU・低容量構成の成功は主張しない。CIと新tag consumerの結果を最後に対応付ける |
| 3 | 公開前確認と変更の統合 | 必要CI、送信全祖先、実payload、公開identity、必要な画像目視 / OCRを確認して、許可済みのmergeを行う。旧providerコピー除去は後回しとし、新規公開物のprivacy gateは維持する |
| 4 | Windows正式SDKを公開する | 版・CHANGELOG・license・取得手順を確定し、固定tagを空consumerへ導入。公開モデルの取得→テキスト / 画像 / 音声・RenderTexture→横断検索→解放・正常終了を確認してReleaseを公開。公開後のtag・asset・手順を読み戻して照合する |

旧providerコピーの除去は未確認のまま別件として保持します。Support案件の読み取りは過去に自動承認レビューで拒否されていますが、今回の計画ではその読み取りも待ち条件にせず、新規メッセージも送信しません。ユーザーの再開指示やprovider側の変更があった場合に別件として扱います。

- 工程4は新CIと公開前確認に絞ります。成功済みの同条件の変換・GPU機能試験は繰り返しません。旧providerコピーの除去は延期であり、現在の送信履歴のprivacy検査とは分けます。
- 重い処理は対応するCI経路が使える範囲で優先します。Unity CIライセンス未設定の同じpreflightを繰り返さず、Personal entitlementをCIへ流用しません。
- macOS / iOS / AndroidはWindows公開後にライセンス・host・端末・署名を確保し、buildと実モデルGPUを別々に受け入れます。端末不足をWindows公開の待ち条件にはしません。
- 実装はTDD・uv・task branch / PRで進めます。公開前にはproject-privacy、versioned Git hooks、全祖先・実payloadを確認し、新規・変更画像は目視とOCRで実bytesを確認します。公開名 `yousan` は維持します。

正式公開日は未確定です。次の見積り更新点は「PRの必要CI完了」「新tag consumerのEditor / Player完了」です。旧providerコピーの対応待ちをリリース日程へ含めません。過去のsource別記録は[検証状況](status.md)と[従来の計画](release-plan.md)に保持します。
