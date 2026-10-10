# 現在の状態と残タスク

確認日: 2026-10-10。最新の確認済み統合mainはPR #25後の `b1a8c67`。統合後main [CI run 38023036522](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38023036522)は全8 job成功。PR #24統合後CI run 38022144437も全8 job成功。下の表はPR #12時点の基準・履歴として保持し、後続実測を本文に追記する。最新CIはGitHubのPR / Actionsを参照。
検索実装は `feat/m2-text-search-sample` / PR #11で統合済み。M1の履歴を維持し、Windows検索実測を別の証拠として追加した。以下のmain SHAとCIは確認時点の基準であり、後続の文書PRでも確認済みの実測SHAを変更しない。

## mainと作業ブランチ

| 対象 | 確認した状態 |
| --- | --- |
| 確認基準main | `4ee0cbbde07fc99ca87d33ed52a44e6100ca0b0c`。PR #1〜#4・#6・#8〜#12統合済み。M1、CI・UPM、検索サンプル・モデル手順・準備最適化と統合記録を反映済み |
| [PR #9](https://github.com/ayutaz/unity-embeddinggemma-2/pull/9) | 計画更新を2026-10-10 02:05:30 JSTにsquash merge。commit `de82a29fc737bca0478c87d06ffcc4d94fb07e64`、統合後CI run 37963795561成功 |
| [PR #10](https://github.com/ayutaz/unity-embeddinggemma-2/pull/10) | 検索基盤を2026-10-10 02:09:08 JSTにsquash merge。commit `be791e00a1b344c5453f70037d7116184169c17f`、統合後CI run 37964228515成功 |
| [PR #11](https://github.com/ayutaz/unity-embeddinggemma-2/pull/11) | 最終head `8c80526`のCI run 37974713580全8 job成功後、2026-10-10 03:41:24 JSTにsquash merge。commit `758cb04`、統合後CI run 37975026704全8 job成功。Windows consumer再現・UI・TDD完了 |
| [PR #12](https://github.com/ayutaz/unity-embeddinggemma-2/pull/12) | 統合後の計画・証跡を2026-10-10 03:47:25 JSTにsquash merge。commit `4ee0cbb`、統合後CI run 37975720534全8 job成功。文書のみでUnityを再実行していない |
| [PR #6](https://github.com/ayutaz/unity-embeddinggemma-2/pull/6) | CI整備・UPM化を2026-10-10 00:07:13 JSTにsquash merge。commit `5ae4e8c29a3a4c8639bd94f7848e3683d063f4d2` |
| [PR #8](https://github.com/ayutaz/unity-embeddinggemma-2/pull/8) | 文書更新を最新mainへ更新し、全8チェック成功後に2026-10-10 00:11:01 JSTにsquash merge。commit `c7d11897e43d374fbb87f3761f65fb3faec60c7c` |
| main保護 | サーバー設定済み。PR必須、strict Required CI（GitHub Actions App 15368）、管理者適用、force push / 削除禁止、linear history・会話解決。人手承認数0 |
| [PR #5](https://github.com/ayutaz/unity-embeddinggemma-2/pull/5) | 文書変更をPR #6へ含め、未マージでクローズ |
| [PR #7](https://github.com/ayutaz/unity-embeddinggemma-2/pull/7) | 文書のみPRのCI確認用。検証後に未マージでクローズ |

全PRを対象とするCI入口、UPMコードと検索APIはmainへ統合済み。サンプルとモデル準備手順はPR #11で追加。mainへの直接pushや保護の迂回は行っていない。[開発規則](../AGENTS.md)に従い、マージは依頼された範囲でCI・差分・競合を確認して行う。

## 検証済みの範囲

| 対象 | 証拠・結果 | 限界 |
| --- | --- | --- |
| M1 | [完了検証](m1-completion-validation.md)。tokenizer全15件一致、fp32 / Float16重みのCPU / GPUCompute各15件、保存・再読み込み・時間 / メモリ測定合格 | Windows Editor 6000.3.16f1 / Sentis 2.6.1、固定モデル・batch 1 / length 128 / 768次元 |
| 確認基準mainのCI | [run 37975720534](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37975720534)。main `4ee0cbb`の全8 job成功。Python4環境、実モデルPython M1 / 検索照合、lint・パッケージ監査・Required CI成功 | Python照合・パッケージ静的監査であり、新たなUnity実行ではない |
| PR #6のCI | [run 37812870591](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37812870591)。Ubuntu / Windows × Python 3.13 / 3.14各102件、lint、実モデルPython15件、パッケージ監査、Required CIの全8 job成功 | 実際のcheckoutは `9ccff937ae9fc675c12aee8e1533d08cc0502e9a`。確認したPR headとtree一致。Python照合はSentis実行ではない |
| 文書のみPR | [run 37813857997](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37813857997)。PR #7はMarkdown 1ファイル・6行追加だけで全8 job成功 | baseは `feat/ci-upm-package`、headは `9633a1ca8d5f3a01eff427599187329c2447b68c`。古いmainへの統合済みという意味ではない |
| UPM導入 | [パッケージ検証](m2-package-validation.md)。別の空のUnityプロジェクトでローカルフォルダ依存解決・compile・契約29件成功、URP依存なし | 当時はモデル不要の導入検証。後続consumerの実モデルGPU結果は別行。Git URL導入は未実行 |
| UPM移行後の回帰 | 元プロジェクトで契約29件・測定契約1件、実モデルM1 3件・API 2件成功、failed / skipped / inconclusive = 0 | 保存・量子化・ベンチマーク全体を移行後に再実行した結果ではない。元M1実測を維持 |
| 検索サンプルのCI | [run 37959339466](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37959339466)。PR #11の実装headで全8 job成功、Python4環境各132件、M1 15入力 + 検索10入力のexport照合合格 | 実checkout `13f194266162d2a728b2b2a0da9e9e58ae314ff6`。Python照合をSentis GPUの成功に数えない |
| Windows検索実モデル | [検索検証記録](m2-search-validation.md) / [全順位と数値](results/m2-search-root-windows-20261010.json)。fp32 / Float16重み × CPU / GPUComputeの4 passed、failed / skipped / inconclusive = 0。固定6文書 / 4queryの全順位一致 | PR側のサンプル、元の検証プロジェクト。他環境や任意入力の品質保証ではない |
| 検索画面操作 | 元プロジェクトGPUとconsumer CPUでGame View入力による準備・日英検索・空入力・解放を確認。元プロジェクトではモデル欠落も確認 | 画面画像と操作記録を保存。任意入力の品質評価ではない |
| 新規consumer | [改善後記録](results/m2-search-consumer-completed-windows-20261010.json)。別checkout / 初期空consumer、導入・監査・変換、sample 18 passed、実モデル4 passed / skip 0、CLI完了応答・全順位・画面操作 | 以前の中断を履歴保持。ローカル依存 / artifactキャッシュは再利用。allocation / fontログの発生元は未確定 |

パッケージは `com.ayutaz.embeddinggemma` / `0.1.0-pre.1`、未リリース。
PR #6のUPM移行時にはRuntimeと契約テストの19ファイルが移行前とGit blob一致し、assembly名・GUIDを維持した。後続の検索API・契約追加は別のTDDと検証記録で扱う。
直接依存はSentis 2.6.1 / Newtonsoft JSON 3.2.2。モデル・ネイティブプラグイン・URP・uloop・Pythonを配布Runtimeへ含めない。

## 残タスクの順序

ユーザーが安定化から正式リリースまでの実行を依頼した。[詳細実行計画](m2-release-plan.md)に4段階の受け渡しと完了条件を記載した。新しい実測はその実行SHA・環境で記録し、以下の既存履歴を書き換えない。

PR #14 / #15は統合済み。main `f564987` の [CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38008701772) は成功した。Git URLは固定commitのPackage Manager解決まで成功し、Editor初回起動は失敗している。Playerの監査済み実モデルbundle準備は成功し、build / 実行は未実行。Windows Editorの完全SHA-256改善は28契約テスト合格、実モデルキャッシュ照合7.38秒の単回測定を確認した。[高速化記録](results/m2-editor-sha256-windows-20261010.json)と詳細計画に現在の証拠を追記し、以下の既存履歴を維持する。

PR #16も統合し、main `0a6e299` の [CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38009534632) 全8 job成功を確認した。続いてPlayer用のhash監査・厳密な参照照合APIをTDDで追加し、合成契約40件が合格した。実行component / build / 実機は未実行。[Player準備](player-validation.md)と詳細計画に受け渡しを記録している。

そのAPIはPR #17で統合し、main `e7b3a54` の [CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38010849703) 全8 job成功を確認した。後続のWindows Playerでは起動・build・測定を接続し、契約65件と実モデル4条件の独立した2回の起動が合格した。M1 15ケース、検索全順位、25入力のtoken一致と終了 / 再起動を確認した。[実測と初回失敗](player-validation.md#windows-player実測-2026-10-10)を参照。Development / Mono2x / stripping Disabled限定で、build警告と終了時のmemory診断は残っている。Git URL consumerと他環境の完了を意味しない。

Windows Playerの変更はPR #18の全8 CI成功後にmain `f44976b` へ統合した。続いて[新しいGit consumerの実動作](m2-git-consumer-runtime-validation.md)を確認し、Git固定SHA解決、compile、UPM 38 / sample 28契約、token15入力と実モデル検索4条件、Game Viewの日英検索 / 空入力 / 解放 / Play停止が合格した。元のconsumer起動失敗は履歴保持。標準layout・project path・SHA等を変えた成功であり、以前の停止原因は確定していない。画面の文字欠けとPersistent allocation Log 1件はリリース前に追跡する。

1. macOS Editor・iOS・Androidのrunner / toolchain / 実機を確保し、精度・backend・速度・メモリを実測する。
2. 測定と並行してhash確認の負荷を改善し、再現しなかったallocationログとEditor font警告を切り分ける。domain reloadの原因は未確定。
3. Git URLのcommit / tag固定導入とサンプル起動、文書・ライセンス・CHANGELOGを確認し、テキスト版をリリースする。ここまででM2完了。
4. M3で画像エンコーダ・GPU画像前処理・テキスト→画像検索を実装・照合する。
5. M4で音声エンコーダ・GPUメル前処理・テキスト→音声検索を実装・照合する。

詳細な依存・受け入れ条件は [M2計画](m2-plan.md)、全体のゴールは [ゴール](goal.md)。

## 未実行と失敗の扱い

- macOS / iOS / Android / WebGPUは未検証。Windows PlayerはMono Developmentと後続IL2CPP / Release / High stripping条件で合格した。ただし検証assemblyのpreserve-all条件で、任意consumerのstrippingを証明しない。Editorや小さいモデルの合格で実機結果を置き換えない。
- クラウドUnityは任意手動Linux CPU検証。Secrets / Variablesは未登録、Editor job未実行。必要なのはこのクラウド経路の利用時で、検索サンプルなどの開発を止める条件ではない。
- PR #6当時のconsumerのuloop launch readinessはタイムアウト。後続run-testsのcompile・29 passedとは分けて履歴を保持する。検索導入では長いパスの失敗後、別の短いconsumerの起動・compile・テストが成功した。
- GPU skipやCPU代替をGPU成功として扱わない。モデルと大きいartifactはGit管理外、LFSを使わずActionsで再生成する。

## 最近の更新

Android用build helperとAPK payload監査を準備した。Windows targetでhelperの9 red、graphics保存リスト復元の2 redを観測し、修正後は既存を含む110 passed。小さいAPK fixtureは16 redから、既存bundleを含む29 passed。その後のUnity 6.3 APIへの置き換えとAndroid target用2契約は再検証待ちで、110件の合格を現在のC#全体へ一般化しない。1回受理されたtarget切り替え後、同じEditor PIDのCLI観測が応答待ちになっている。追加起動・再起動はしていない。実APK build / 実機は未実行で、draft PRとして通常CIと分けて追跡する。[準備状況と再開手順](m2-android-validation.md)。

PR #24統合main `01f3d86`を新しい空のGit consumerへ固定導入し、requested / resolved SHA・88ファイル・import済みsampleを照合した。compile、UPM 38 / sample EditMode 64、token15入力、実モデル検索FP32 / Float16 × CPU / GPUの4条件と全順位・scoreが合格。sample PlayMode 2件は受理後のCLI接続失敗を保持し、同じ実行のUnity保存XMLから2 passedを回収した。Game Viewの日英入力・準備・検索・空入力・解放・warm完全hash・Play停止も確認した。[新規Git導入の結果](m2-latest-git-consumer-validation.md)。旧Editorの終了を確認してから新Editor1つを起動し、再起動なし。停止時Persistent allocation Log 1件の原因、実APK / 他OS / 実機 / tag導入と正式Release gateは残る。

配布TextSearch sampleの選択モデル配置と所有cacheを実装し、実行source `2e463a8`でEditMode 64 / PlayMode 2 / uv Python 22件が合格した。Windows Editorの実file URL転送・FP32 / Float16のactual GPU準備・固定日英2queryの全順位とscore・空入力・解放・欠落モデル拒否・実worker使用中のleaseを確認した。warmではmanifestだけを転送しモデル再転送0、完全CNG監査約803ms / 418ms、成功後は選択Float16 1組の約586MBだけを保持した。[実測と制約](m2-sample-cache-validation.md)。stage時間はmodel deserialize / 文書推論を含まない。変更sample SHAの新規Git導入、実APK / 実機と新sample Player画面は未実行、font Warning 1件の由来も未確定。次は新規固定Git導入と実APK / 他環境のgateを進める。

Windows IL2CPP / compiler Release / managed stripping Highのbuildと実モデル2起動を確認した。release build helperの3 red → 99 green、build 257.1秒 / errors 0 / warnings 485、両Player終了code 0。4条件のactual CPU / GPU、25入力token一致、M1・検索全順位・score・解放が合格。file URLの初回6件展開から2回目manifest 1件だけのcache再利用へ移行し、実CNGの完全hash監査1.47秒 / 1.55秒、モデル再転送0、保持1 bundleを確認した。[結果と範囲](m2-player-il2cpp-validation.md)。検証assemblyにpreserve-allを指定した条件で、任意consumerのstripping・他OS・native leak解消を証明した結果ではない。次は配布sampleのAndroid配置、実APK / 他OS / 実機、残る安定性とrelease gateを進める。

PlayerのURL配置は、完全SHA-256を毎回照合する所有cacheへ改善した。cacheのred / greenとWindows CNGのred / green、全96契約が合格。同じ約1.68GBの実bundleでwarm監査3回が合格し、モデル再転送0、監査約1.5〜1.8秒を確認した。[測定と限界](m2-bundle-cache-validation.md)にbaseline約108秒の観測とNUnit timeout失敗を保存する。保持modelは成功後1 bundle、更新中最大2 bundle。後続のWindows Player / IL2CPP結果は上記に追加し、他OS・実機 / sample配置と安定性・正式releaseのgateは残る。

Player用のStreamingAssets展開adapterは新規12件のredから、既存を含む77件のgreenへ移行した。Windows Editorの実UnityWebRequestによる小さいfile URL転送と欠落処理を確認し、jar transportは注入して検証した。[配置経路と残る条件](player-validation.md#streamingassetsからの展開adapter)を参照。展開後も完全SHA-256・token・全backend条件を要求する。後続のWindows実Player / cache確認は上記に追加した。実APK / Android端末 / 配布sample配置は未実行。環境inventoryではGitHub Secretとrunnerは0、ADB接続端末も0だった。他OS / 実機の合格や正式releaseを完了とはしない。

検索UIの文字欠けは、実OS fontのglyph境界を使うTDDで再現し、font・sizeを明示したcached styleへ修正した。EditMode 34件とPlayMode 1件が合格し、Windows Game Viewの日英GPU検索の全順位・scoreが修正前と完全一致、空入力・解放・モデル欠落表示・Play停止も確認した。[表示修正の実測](m2-ui-font-validation.md)を参照。修正したsampleはGit consumerへ作業ソースとして配置したもので、修正SHAの新規Git導入成功とは扱わない。allocationとdomain reloadの原因、他OS / 実機と正式リリースの条件は未達のまま残す。Windows IL2CPP / release strippingの後続結果は上記を参照。

## 文書の読み方

README、ゴール、M2計画、CI、API、automationは現在の利用・作業手順。
M1の各plan / validationは実行した段階の履歴で、当時のcommit・数値・失敗を維持する。
[技術調査](technical-approach.md)は採用済みのテキスト経路と未実装の画像 / 音声設計を区別する。
[新規性調査](embeddinggemma-2-unity-novelty.md)は2026-10-07の調査に2026-10-09の限定検索を追記したもので、先行実装の不存在を保証しない。
package内の文書は導入 / API / 未リリースと、mainへ統合したサンプルの使い方を説明する。AGENTS.mdとApacheライセンスは規則・条文を維持する。
