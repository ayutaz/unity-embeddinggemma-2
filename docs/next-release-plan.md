# Windows先行リリースまでの実行計画

更新: 2026-10-20。ユーザーの方針により、最初の正式リリースはWindows Editor / Playerを対象にします。macOS / iOS / Androidは今後の対応です。[ゴール](goal.md)のテキスト・画像・音声、横断検索、FP32 / 採用Float16重み、Sentis GPU、数値・運用・性能・privacy・新規導入の条件は維持します。

**工程1〜3のWindows機能検証は完了しました。** [確認記録](results/windows-first-stages-1-to-3-20261020.json)に実行sourceと適用範囲を記録しています。残る工程は4「性能・最低要件・公開前整理」と5「新tag導入・正式公開」です。正式リリース完了の報告ではありません。

## 現在地

公開mainは `1a036fb`（[#113](https://github.com/ayutaz/unity-embeddinggemma-2/pull/113)まで統合済み）。head `c980027` の必須CI全15 job、実13数値archive、元824,320値とCPU encoder 120 capture / 30,822,400値の独立照合、実mainの同tree・署名・公開identity・全祖先を確認しました。許容差超過は0件です。初回upload timeoutとその失敗記録を保持し、同sourceの限定再試行を受け入れています。

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

元の最終vector・中間値の参照と許容差は変更していません。既存の音声GPU4条件16回の[測定](results/audio-gpu-repeat-four-conditions-20261020.json)は保持しますが、同条件CPU比較・全区間のメモリ観測・最低要件は未完了です。今回の機能検証のprocess時間を推論速度やGPU専有時間として扱いません。新tag consumer、正式版・正式Release、他hardware / platformの合格も未完了です。

## 次の進め方

工程4の開始点は、同一host音声CPU基準の[準備停止（source `8f1bb97`、exit124）](results/windows-cpu-baseline-preparation-stop-20261020.json)です。非terminalのまま残った親receiptと、実processのterminal記録を区別し、この比較を実行中・合格とは扱いません。

1. 元ログ・source・既存成果物・上限を一度診断し、再利用できるbytesと不足する処理を確定します。参照や許容差を変更せず、新しい根拠のない同じ準備を繰り返しません。
2. 数値gateと正常終了を満たした条件だけで、同一入力・profile・仕事量のCPU / GPU、初回 / warm・各stageを比較します。別hostのCI数値合格と、このhostの性能は別の欄に記録します。
3. RAM / VRAMの観測範囲、実測した対応構成と余裕を明記して要件を定義します。未測定peakや低い構成の成功を推測せず、追加機材の購入・他端末の確保をWindows先行計画の前提にはしません。
4. 現行履歴・実公開payloadのprivacyと旧providerコピーの除去確認をそれぞれ閉じ、工程5へ進めます。旧provider対応は返答・状態変更時と公開判定前に確認します。

- 工程4は不足する比較と公開前確認に絞ります。成功済みの同条件の変換・GPU機能試験は繰り返しません。旧providerコピーは現在の履歴の安全性と別に除去を確認します。
- 重い処理は対応するCI経路が使える範囲で優先します。Unity CIライセンス未設定の同じpreflightを繰り返さず、Personal entitlementをCIへ流用しません。
- macOS / iOS / AndroidはWindows公開後にライセンス・host・端末・署名を確保し、buildと実モデルGPUを別々に受け入れます。端末不足をWindows公開の待ち条件にはしません。
- 実装はTDD・uv・task branch / PRで進めます。公開前にはproject-privacy、versioned Git hooks、全祖先・実payloadを確認し、新規・変更画像は目視とOCRで実bytesを確認します。公開名 `yousan` は維持します。

正式公開日は未確定です。次の見積り更新点は「同条件性能比較・最低要件の確定」「公開前整理の完了」「新tag consumerの完了」です。開発作業とprovider対応の待ち時間を分けて報告します。過去のsource別記録は[検証状況](status.md)と[従来の計画](release-plan.md)に保持します。
