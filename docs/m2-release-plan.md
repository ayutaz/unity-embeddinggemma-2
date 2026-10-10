# M2: 安定化からテキスト版リリースまでの実行計画

作成日: 2026-10-10。開始時点のmainは `2329507eb763202a297c1b15f385a83512eb6e1f`。
ユーザーの「4まで詳細の計画を立てて進めてください」に対応し、安定性・高速化、Git URL導入、他環境の実測、リリースの4段階を扱う。
M1の履歴と既存のM2受け入れ条件は維持する。Windowsだけの成功でM2全体を完了としない。

## 開始時点の証拠

- 最新mainのWindows consumerでUnity 6000.3.16f1 / Sentis 2.6.1 / uloop-cli 3.8.1を実行した。実モデルFP32 / Float16重み × CPU / GPUComputeは4 passed、failed / skipped / inconclusive = 0。固定6文書 / 4queryの全順位がPython参照と一致した。
- Game Viewへ実際のmouse / keyboardイベントを送り、GPUで日本語 / 英語検索、空入力の拒否、モデル解放、Play停止を確認した。新しい結果は `artifacts/unity-harness/main-2329507-live-20261010/verification.json` と同ディレクトリの数値・操作・画像に保存。大きいモデルは保存しない。
- Play終了時のPersistent allocation Logは1件。スタック付きで同じ操作を繰り返すと再現せず、`Deleting invalid font reference` Warningが1件出た。解消済みではない。
- 同consumerの `SHA256.Create()` は `SHA256Managed`。遅いhash確認の候補であり、domain reloadの原因を確定した証拠ではない。
- GitHub APIでself-hosted runnerは0件、Repository Secrets / Variablesは0件。WindowsにはAndroid / Windows Standalone / WebGL modulesがあり、ADB接続端末は0件。Mac / iOS / Android実機の利用可否は確認待ち。

## 1. 安定性と高速化

### 作業

1. Play開始前、準備後、検索後、解放後、Play終了後のログとリソース状態を保存する。モデル不要の空scene、サンプルUIのみ、CPU、GPUの順で条件を限定する。NativeLeakDetectionの設定は検証後に元へ戻す。
2. allocationログとfont警告を別々に扱う。Editor / uloop / Sentis / サンプルのどこで発生するかを、スタックと最小再現で特定する。再現したものは先に失敗テストを作り、修正後に同じ経路を通して確認する。再現しないものを修正済みと書かない。
3. hash処理、モデルdeserialize、保存 / 量子化、import、compile / domain reloadを別々に計測する。元のモデル、FP32 / Float16 / tokenizerの完全なSHA-256照合は維持する。長さやmtimeだけを完全な検証の代用にしない。
4. 小さいデータでhash実装の正しさと性能を測り、有効な変更だけを実ファイルで比較する。既存のキャッシュ再利用と必要箇所だけのimportを維持する。失敗時は部分成果物を成功として再利用しない。
5. 初回ロード、初回推論、warmup後推論、解放を計測する。Editor全体のメモリと推論器の観測値を区別し、測定できないGPUメモリはunknownとする。

### 成果物と合格条件

- 再現手順、変更前後の条件別ログ、必要な回帰テスト、CPU / GPUの時間・メモリ要約。
- 修正した問題は意図したred → greenと実際のPlay操作で確認する。未確定の問題は追跡項目を残す。
- 性能変更は同じモデルhash / Editor / backend / 測定条件で比較する。単回測定は単回と明記し、精度4条件・全順位・解放が退行していないことを確認する。

## 2. commit固定Git URLからの導入

### 作業

1. mainの固定40桁SHAを指定し、新しい短いパスの空consumerを作る。`create_consumer --git-revision <SHA> --automation --sample` を使い、packageはGit URL、uloopはEditor側だけに置く。
2. Unity Package Managerの解決結果を調べ、requested revision、resolved hash、package source、直接・推移依存を記録する。ローカルfile依存の結果をGit導入成功の代用にしない。
3. Runtime / sample / assembly / GUIDを確認し、compile・package契約・sample契約を実Editorで実行する。URPやネイティブプラグインの不要な依存がないことを確認する。
4. 固定revisionの既存監査済みモデルを別途配置し、同じPython参照で実モデル4条件を検証する。cacheを使った場合はその範囲を記録し、新規生成や空cache成功とは表現しない。
5. Game Viewから準備 → 日本語検索 → 英語検索 → 空入力 → 解放 → Play停止まで操作する。モデル欠落も確認する。
6. これを利用者が再現できる手順として文書化する。候補commitが変われば配布コードの差分を確認し、必要な導入・回帰だけ再実行する。

### 成果物と合格条件

- consumer manifest / packages-lock、Git解決hash、compile・契約・実モデル結果、操作記録、小さいスクリーンショット。
- Git URL依存のままで固定SHAのコードを解決し、sampleと実モデルを実行できる。modelと大きい成果物はGitに含めない。

## 3. macOS / iOS / AndroidとPlayerの実測

### 環境確保

| 対象 | 必要な環境 | 現状 / 次の作業 |
| --- | --- | --- |
| Windows Player | Windows build module、実GPU、固定モデル配置 | moduleあり。Editor成功と別にbuild / Player起動を検証する |
| macOS Editor | Mac、Unity 6000.3.16f1、Metal対応GPU、適切なUnityライセンス | 実行可能なMacを確認する。hosted CPU buildだけでMetal GPU成功としない |
| iOS Player | Mac、Xcode / iOS module、署名可能な実機、Metal | 端末・署名・接続を確認する。Apple資格情報はチャットやソースへ記録しない |
| Android Player | Android module / SDK / NDK / JDK、実機、対応graphics API | Windows moduleあり、接続端末なし。ABI / API / メモリ条件を確認する |
| Actions Unity | 適切なライセンス、対象OS / GPU / 接続端末を持つrunner | Secrets / self-hosted runnerなし。利用可能な構成を先に確定する |

### 実装・実行

1. Playerのモデル配置・読み込み経路をTDDで整える。AndroidのStreamingAssets等、通常のファイルパスとして読めない配置を実際の対象で確認する。変更前に対応する失敗を検証する。
2. 固定参照と同じprompt / tokenizer / batch 1 / length 128を使う実行ハーネスを用意する。requested backendとactual backendを記録し、GPU非対応・OOM・build失敗を別の失敗として保存する。
3. 各対象でtoken ID / mask 15ケース完全一致、有限768次元・単位長、FP32 cosine >= 0.999 / Float16重み >= 0.99、検索6文書 / 4queryの全順位一致を確認する。
4. load、初回推論、warmup後の反復推論、終了 / 再ロードを測る。OS / Unity / GPU / graphics API / ABI / build backend / stripping / メモリ測定API / モデルhash / commitを結果へ記録する。
5. build・モデル取得・参照生成・可能なテストはActionsへ寄せる。実機GPUは利用可能な接続端末か信頼された専用runnerで実行する。fork PRへSecretsを渡さない。
6. 結果収集はテスト・benchmarkの出力に限定する。利用者の入力収集や常駐サービスは追加しない。

### 成果物と合格条件

- 環境inventory、build / 実行手順、機械で監査可能な結果JSON、各環境の精度・時間・メモリ表、失敗の追跡記録。
- macOS Editor / iOS / Androidの実モデルCPU / GPUを実行して必要な条件を満たす。GPU skip / CPU fallbackをGPU passにしない。
- 環境が確保できない場合は未実行のまま残す。この条件が残った状態でM2全体を完了・リリース済みとしない。

## 4. テキスト版のリリース

### 作業

1. 1〜3の結果を候補commitへ対応付ける。対象環境の未実行、必要な失敗、サンプルの未解決問題を点検し、変更を凍結した候補で監査する。
2. README、UPM文書、CHANGELOG、Apache-2.0のコードライセンスとモデルの取得・利用条件を確認する。コードのライセンスをモデルへ自動適用しない。モデルを配布artifactやGitへ混入させない。
3. リリース版versionとcandidateをPRで更新し、Required CIと必要なUnity回帰を確認する。mainへ直接pushしない。
4. tagとGitHub Releaseは検証済みcommitに対応させる。tag固定Git URLで空consumerへ導入し、compile・サンプル・実モデルを確認する。検証前に正式公開済みと表現しない。
5. Release notesへ実測した対応環境、固定モデルrevision、API制約、モデル取得手順、既知の制約を記載し、公開後のtag / version / Releaseと手順を読み戻して確認する。

### 合格条件

- 固定tagから再現できるUPMパッケージとサンプルがあり、必須環境の実測結果・文書・CHANGELOG・ライセンスが揃っている。
- すべてのgateを監査し、未実行を成功に数えず、GitHub Releaseとtagの実在を確認してM2完了とする。

## PR分割と依存

| PR | 範囲 | 次へ渡す証拠 |
| --- | --- | --- |
| A | 本詳細計画、環境inventory、Git導入の実測と必要な監査改善 | 固定Git SHA導入の結果、環境不足一覧 |
| B | 安定性・hash処理の測定 / 改善 | red / green、同条件性能比較、Unity精度・UI回帰 |
| C | Playerの配置・実行・測定ハーネス | 小さい契約テスト、build / 実行結果schema、Windows Player検証 |
| D | macOS / iOS / Androidの実測と必要な移植修正 | 各環境の厳密な結果、build・端末・backend条件 |
| E | リリース版・文書・tag導入・配布 | 1〜3の合格を揃えたcandidate、tag導入結果、Release |

AとBは環境調査と並行して進める。Cは実機の有無に依存せず必要な準備を進める。Dの実測は実行環境に依存する。Eの正式リリースは必要環境の合格に依存する。
Pythonは `tools/` とuvに統一する。実装はred → greenを確認し、小さい必要チェック以外はActionsを優先する。モデル・大きいartifactはGitに含めない。
PRのmergeはユーザーの既存依頼の範囲で、差分・競合・現在headのCIを確認して行う。リリースまでを依頼された本作業の範囲で進め、未達gateを迂回しない。

## 実行中の記録

- PR #14で本計画とGit導入監査を進めている。新しい `artifacts/g` はローカルfile依存を使わず、commit `2329507eb763202a297c1b15f385a83512eb6e1f` を指定した。
- Package Managerの `packages-lock.json` は `source: git` と同じ40桁hashを記録した。Sentis 2.6.1 / Newtonsoft JSON 3.2.2も解決した。[部分成功と起動失敗の記録](results/m2-git-consumer-windows-20261010.json)を参照。Git解決の監査CLIをTDDで追加し、既存packageテストを含む38件が合格した。
- 初回launchのCLI readinessは330秒でtimeout。後続get-logsは `UNITY_EDITOR_UNRESPONSIVE`、IPC heartbeatは生存しEditor main threadのtickが停止していると報告した。compile / 契約 / 実モデル / UIは未実行であり、Git URLでの全動作確認は未完了。
- 旧consumer `artifacts/c` はPlay停止・scene cleanを確認して閉じた。終了指示だけでは旧Editorが残ったため、次回からprocess終了を確認してから次を起動する。新Editorをtimeoutだけを理由に再起動しない。
- PR #14は現在headの全8 CI成功後に統合した。次は [Player向け準備](player-validation.md) のモデル・参照bundleを整備した。fixtureのred → green、関連70テスト合格、既存実モデル約1.68GBのcopy前後完全hash一致を確認した。Player build / 実行の合格とは別の結果として扱う。
- PR #15も全8 CI成功後に統合し、main `f564987` の [CI run 38008701772](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38008701772) が成功した。
- Git consumer `artifacts/g` の起動失敗を記録して終了し、processの終了を確認してから既存consumer `artifacts/c` を1つだけ起動した。別の安定性・hash改善の検証に利用しており、Git URLでの実行成功には数えない。
- Windows Editorの完全SHA-256を高速化した。新規10テストの意図したred → green、既存18件を含む28 passedを確認。実モデルのキャッシュ再利用はEditor内7.38秒の単回測定、receiptとモデルの更新時刻は不変だった。[測定条件・回帰・限界](results/m2-editor-sha256-windows-20261010.json)を参照。domain reloadとallocation / font警告の原因は未確定のまま扱う。
- PR #16を全8 CI成功後に統合し、main `0a6e299` の [CI run 38009534632](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38009534632) 全8 job成功を確認した。
- 同じEditorでモデルを含まない空sceneと、モデルを準備しないsample UIを比較した。空sceneのPlay終了でPersistent allocation Logが1件、sample UIではfont生成を確認しLog / Warning / Errorは0件だった。先行する実モデル実行と全Editor依存が同じprocessにあるため、由来を確定した結果ではない。元scene・Play停止・NativeLeakDetection設定を復元した。[切り分け記録](results/m2-stability-baseline-windows-20261010.json)を参照。
- Player専用assemblyに参照・token・vector・全順位・backend・解放・完了の検証APIと完全hash loaderを追加した。意図したredを段階的に観測し、合成参照を使う契約40件が合格した。[準備状況](player-validation.md)に次の起動component / build helper / Windows Player実行とAndroid展開を記載する。実モデル / GPU / Playerの新規成功には数えない。
- PR #17を統合し、main `e7b3a54` の [CI run 38010849703](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38010849703) 全8 job成功を確認した。
- Windows Playerの起動・build・測定を接続した。追加実装と結果保存の競合修正のred → green、契約65 passedを確認。commit `dfdcc9c` のDevelopment / Mono2x / stripping Disabled Playerを2回起動し、終了code 0、実モデルCPU / GPUCompute × FP32 / Float16の4条件、25入力のtoken ID / mask一致、M1 15ケース・検索全順位・解放が合格した。[結果](results/m2-player-windows-20261010.json)に初回Player失敗とbuild警告・終了時メモリ診断も保持する。Windows IL2CPP / release stripping、Git URL全動作、macOS Editor / iOS / Android、安定性の追跡とリリースgateは残る。
- PR #18を全8 CI成功後に統合し、ローカルmainも `f44976b` へfast-forwardした。既存のUnity設定変更を保持した。統合後 [CI run 38013974138](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38013974138) は記録時点で実行中。
- 新しい空のGit consumer `artifacts/h` で固定main `e7b3a54` のGit解決・初期化・compile・UPM契約38件・sample契約28件・15 tokenケースと実モデル検索4条件・Game View日英入力 / 空入力 / 解放 / Play停止を確認した。[実動作の記録](m2-git-consumer-runtime-validation.md)。最新mainのUPM tree一致も確認した。以前の初期化失敗を保持し、今回は標準layoutと1 Editor運用を含む条件変更で成功した。停止原因の特定ではない。
- 画面確認でlabel / 結果本文の縦方向の文字欠けを確認した。font metricsとGUILayout高さの回帰修正をリリース前の作業へ追加する。Persistent allocation Log 1件も未解決。次はこれらの安定化とAndroid配置adapter、Windows IL2CPP条件、他OS / 実機の環境確保を進める。正式リリースgateは維持する。
- PR #19を現在headの全8 CI成功後に統合し、mainを `e958d39` へfast-forwardした。PR #18統合後main `f44976b` のCI run 38013974138も全8 job成功を確認した。既存のUnity設定変更は保持した。
- font未指定styleの高さと実OS fontのglyph境界の不一致を新規テストで再現し、sampleのGUIStyleを実font・size指定のcacheへ変更した。意図した5 failed / 1 passedから、EditMode 34 passed + PlayMode 1 passedへ移行。Windows Game Viewで日英GPU検索の全順位・score不変、空入力拒否・解放・モデル欠落表示・Play停止を確認した。[修正と検証範囲](m2-ui-font-validation.md)を参照。Git consumerのimport済みsampleを作業ソースへ置き換えた結果であり、修正commitの新規Git導入・他OS表示・allocation修正・性能改善の証拠にはしない。残る安定性・Android配置adapter・IL2CPP・実機環境・リリースgateを引き続き進める。
