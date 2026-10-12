# Unity 6000.3.21f1更新と終了比較

2026-10-12にユーザーが6000.3.21f1への固定版更新を承認した。公式revisionは`c02631ffc030`。関連する[公式修正UUM-146676](https://unity.com/releases/editor/whats-new/6000.3.21f1)はWindows accessibilityの終了クラッシュであり、今回の問題との一致は実行比較で判断する。

## 更新範囲

- 作業ブランチのProjectVersion、空consumer生成、手動Unity CPU CIの実行版を21f1にした。
- 新しいPlayer build provenanceは21f1を必須にする。過去16f1の保存結果は書き換えない。
- Sentisは2.6.1のまま。既存の監査済みモデル、Python参照、Library / build cacheを再利用する。
- 配布packageのminimum 16f1は以前のEditor検証に対応した値。21f1のnative検証は下記の補修済みconsumer条件で確認した。標準の新規導入と配布条件は別gateで確認する。
- Androidの過去APK / bundle監査は16f1 provenanceのまま保持する。21f1のAndroid実機合格へ読み替えない。

## TDDと導入

版指定Python契約は2 failedを確認してから実装し、関連30件がpassed。Player provenanceの新21f1受け入れ・旧16f1拒否は2 failed → 2 passedを確認した。この2件は旧Editor上のmetadata契約で、実21f1での起動や精度の証明ではない。

公式Windows Editor / IL2CPP installerを取得し、Authenticode Valid / Unity Technologies SF署名を確認した。Editorはworkspace内の専用ディレクトリに導入し、実ファイルのProductVersion `6000.3.21f1_c02631ffc030`を確認。IL2CPP moduleとnative回帰の完了は下のgateで別途確認する。installer・Editor本体・生ログはGitに入れない。

## 実行順序とgate

1. 保存sceneがclean・Play / compile / Test Runner停止を確認し、旧Editorを正常終了する。終了後のPID消失を確認してから21f1を1つだけ起動する。
2. 同じconsumerをuloopでcompile・契約確認する。実行中のoperation / PID / 保存結果を継続観測し、観測timeoutだけで再実行しない。
3. model-freeのnative終了を比較する。旧通常モードの終了アクセス違反と、同じ実行経路での21f1終了を分けて記録する。モデル欠落の想定exit 1を実モデル合格と数えない。
4. sample IL2CPP / Release / High strippingで日本語・英語GUI、全4query / 各6順位・score、空入力・欠落モデル・解放を固定Python参照と照合する。独立2起動がexit 0で、初回 / warm cacheの監査も通ることを必要条件にする。
5. 新固定Git candidate導入と実モデルCPU / GPU × FP32 / Float16の回帰を必要範囲で確認する。Android / Mac / iOS実測と正式Releaseの条件も維持する。

## 21f1 Editor移行で確認した結果

公式Windows IL2CPP moduleをworkspace内のEditorに導入し、win64のRelease / development IL2CPP variationsを確認した。旧16f1 EditorはPlay / compile / Test Runner停止、scene cleanを確認して正常終了し、そのPID消失後に新版を起動した。

最初の起動は`ALLUSERSPROFILE`欠落でPackage Managerがpath undefinedとなった。起動環境に`ProgramData`の値を補って解決した。この失敗をcompile合格には数えない。

続いて21f1同梱Test Framework 1.6.0（consumer fingerprint `bd7f943e9647`）のC# 2ファイルに.metaがなく、対応typeがcompile対象に入らなかった。検証consumerのignored PackageCache内で`EnumerableApplyChangesToContextCommand.cs`と`PostbuildCleanupWithJobDataAttributeFinder.cs`の.metaだけを補った。C#ソースは変更していない。現在のEditor結果はこの環境補修を前提とし、未補修の新規21f1導入がそのまま動く証拠ではない。初期refreshは70.053秒、補修後compileは成功した。これは高速化比較の合格ではない。

新21f1 preparationを監査・stageする契約を追加し、実際のorigin版を保持する。旧16f1と新21f1だけを許可し、別版・nullを拒否する。Pythonは1 failed → 18 passed、C# loaderは21f1 originの1 failedを確認後に対応した。21f1でのvalidation assembly回帰は151件中149 passed / 2 failed / skip 0。2件はAndroid targetで実行するbuild integration契約をWindows targetで実行した前提不一致で、Android合格には数えない。bundle loader全15件と新Player provenance契約はこの149件に含まれる。

更新checkpoint `d6d027e`の[CI 38187716032](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38187716032)は全8 job成功。CIを21f1 native実行の証拠には数えない。

## Windows nativeの終了比較

固定source `725200ad03036f8ac9ab31f31007987c1f66bb06`で21f1 / Sentis 2.6.1 / IL2CPP / Release / High strippingの精度Playerを実buildした。receiptはSucceeded、410.119秒、errors 3 / warnings 485。sample PlayerはSucceeded、115.387秒、errors 0 / warnings 485。warning / errorがゼロのbuildとして扱わない。精度Playerの18件、sampleの37件の記録ソースhashは作業ブランチと一致した。consumerのGit依存は既存の`fc66af7`で、これを新candidateの空consumer Git導入完了とは数えない。

通常モード / Direct3D12 / Hiddenのモデル欠落経路は、runtimeがDirectoryNotFoundのfailedを保存した後、PIDの実終了codeが想定どおり1になった（38.079秒）。旧16f1でアクセス違反になった同経路において、今回はnative crashを観測しなかった。実モデル・GPU合格とは別の終了比較である。

通常モードの実モデル精度Playerは独立2起動とも、token 25行、CPU / GPUCompute × FP32 / Float16の4条件、M1 15入力・6文書・4query / 各6順位の固定Python参照照合に合格し、実終了code 0になった。NVIDIA GeForce RTX 4070 Ti SUPER / Direct3D12。GPU skipやinjected providerは使っていない。元の16f1 preparation originを保持した監査済みbundleでの実行であり、初回はEditorのimport / sample buildと重なったため速度比較には使わない。1回目563.843秒、Editorを閉じて開始した2回目363.159秒。これは同条件の高速化試験ではない。

21f1で監査済み元.pt2を実際にimport・FP32 / Float16で保存し、新しいpreparationとbundleを作成した。FP32 / Float16 / tokenizerの各bytes・SHA-256は旧16f1保存物と同一だった。準備完了をGPU実行の証拠には数えない。

Float16/GPUの長時間化は未解決。1回目のwarm 3回は8〜17秒、FP32/GPUは約150msだった。Editorを閉じた2回目もFloat16/GPU 7.354 / 8.414 / 7.389秒に対してFP32/GPU 107.480 / 120.763 / 100.755msであり、Editorだけが原因とは扱えない。Editor / import workerを正常終了してから以後のPlayerを順に実行する。VRAM使用量約14GB/16GBも観測したが、WDDMではprocess別使用量を取得できず、原因をEditorや特定allocatorに断定しない。4条件を同一frameで同期実行するため、条件間でframeを返してGPU資源の解放を待つ案を次のTDD / 実測候補にする。これは未検証仮説である。sampleの既存defaultはFP32で維持する。

sampleのFloat16指定baselineはcold / warmとも実モデルGPU・4query / 各6順位・score・空入力拒否・モデル欠落拒否・Worker / font解放に合格し、実終了code 0だった。各110.615 / 108.706秒。cacheはcoldが3ファイル転送、warmが1ファイル（descriptorのみ）転送、モデル再転送なし。日英の1280×800画像を確認し、batchmodeではない通常GUIで文字と結果が描画された。これはGUI APIを通るハーネスの結果であり、native keyboard / mouse入力の人手操作を確認した結果ではない。

製品sampleの既定ModelPathはFP32だったが、`PlayerSampleView.Prepare`が検証時にFloat16へ変更していた。proxyの準備入力をFP32 / GPUComputeへ揃える契約は1 failedを確認後に実装し、batchmode / nographicsのsample契約13件がpassed / skip 0になった。画面Editorの再起動はengine初期化前で待機し、テスト未実行だったため、GUI不要の契約はbatch Test Runnerへ切り分けた。原因は未確定で、domain reloadの合格やGPU合格に読み替えない。

checkpoint `cdd7c30`の初回FP32 Playerは全query・解放・終了code 0、31.112秒。保存scopeに旧Float16表記が残っていたため、この結果を最終受け入れの2起動には使わず保持した。表記の契約は1 failed → 実行契約全22 passed / skip 0で修正した。`cdd7c30`の[CI 38191479251](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38191479251)は全8 job成功。

## FP32既定sampleの最終native結果

固定source `28529f2e20a57dc865ef7268a92b0526c9a87c3e`でsampleを実buildし、全37 source hash一致、Succeeded / 76.319秒 / errors 0 / warnings 485を確認した。GUI不要のbuildはbatchmode / nographicsで実行し、以下のPlayerはbatchmodeなしの通常表示 / Direct3D12で実行した。Editorの実終了後にPlayerを直列で起動し、既存Library / build cacheと監査済み21f1 bundleを再利用した。モデル再downloadは行っていない。

| 測定 | cold | warm |
| --- | --- | --- |
| 実process終了code | 0 | 0 |
| process全体 | 31.571秒 | 29.187秒 |
| 準備（stage / audit / load / Worker / 6文書埋め込みの合算） | 11.557秒 | 8.237秒 |
| 全4queryの検索時間 | 149.932〜207.101ms | 263.912〜329.933ms |
| cache再利用 / 転送数 | false / 3 | true / 1（descriptorのみ） |
| 固定6順位・score / 最大score差 | 全一致 / 9.187e-8 | 全一致 / 9.187e-8 |

両方でFP32 / actual GPUCompute・空入力拒否・モデル欠落拒否・Workerとfont解放、Console 0を確認した。既存uv監査CLIも両process gateを合格と判定した。保存した全順位とscoreを固定参照へ別途再照合し、画像のbytes / hashも照合した。coldの日英1280×800画像を目視確認し、文字と結果、FP32 path表示を確認した。native keyboard / mouseの人手操作結果ではない。

Float16指定baselineの110.615 / 108.706秒からFP32指定では31.571 / 29.187秒になった。これは既定モデルに合わせた検証条件の変更であり、Float16の同条件高速化やdomain reload高速化の証明ではない。warm検索がcoldより遅い観測も保存し、反復benchmarkの中央値や保証値にはしない。

公開用の小さい[数値・source・条件別要約](results/m2-unity21-windows-20261012.json)を保存した。生ログ、ローカルpath入りPNG、モデル・binary・dumpはignored artifactsに残しGitへ含めない。

新規candidate導入、未補修21f1環境のcompile、Float16/GPU性能調査、他環境は未完了。PR #31をdraftのまま進める。
