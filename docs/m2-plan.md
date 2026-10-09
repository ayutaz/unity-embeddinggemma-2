# M2 詳細計画と残作業

更新日: 2026-10-10。基準main: `be791e00a1b344c5453f70037d7116184169c17f`（PR #10統合後）。
**CI整備・UPM化はPR #6でmainへ統合済み。統合後のmain CIも全8 job成功。** PR #1〜#4のWindows EditorのM1は完了した。
GitHub側のmain保護も設定・再確認済み。検索基盤はPR #10でmainへ統合済み。検索サンプルはPR #11で実装し、Windows実モデル4条件と元プロジェクトの画面操作を確認した。新規consumer再現の完了応答・画面操作、他環境・リリースは残る。M2全体は進行中。[パッケージ検証](m2-package-validation.md)を参照。
[現在の状態](status.md)でPRと証拠の対応を確認し、[M1完了検証](m1-completion-validation.md)を既存の基準としてテキスト検索サンプル、モデル準備手順、他環境検証、配布を進める。

利用者向け一覧の1（検索サンプル）と2（モデル準備手順）を実装中。[個別の実装計画](m2-search-plan.md) / [検証記録](m2-search-validation.md) / [モデル準備手順](model-preparation.md)を参照。
サンプルのEditMode / PlayMode契約と、Sentis実モデル4条件（fp32 / Float16重み × CPU / GPUCompute、全queryの全順位）が合格した。新規consumerの数値照合も保存済みだが、テスト後のdomain reload中に停止されCLI完了応答は未取得。高速化の検証とconsumer画面操作が揃うまで1 / 2の作業完了とはしない。PR #9 / #10は統合済み、#11は未統合。

## 現在地

- Unity 6000.3.16f1 / Sentis 2.6.1。PR #6で公開APIを `Packages/com.ayutaz.embeddinggemma/Runtime/` へ移行。開発版 `0.1.0-pre.1`、未リリース。
- `TextEmbedder` はquery / document / raw、同期・メインスレッド、batch 1 / length 128 / 768次元、CPU / GPUComputeを扱う。
- `TextModelFile` はfp32 / Float16重みの保存・再読み込みを実装済み。Windows Editorで両形式・両backend全15件が合格。
- Runtime asmdefはSentisと `Unity.Newtonsoft.Json` を参照する。現在のlockでNewtonsoftは3.2.2。uloop、URP、2D関連パッケージは検証プロジェクト側の構成であり、配布Runtimeの依存には持ち込まない。
- 現在のmain CI [run 37964228515](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37964228515)は全8 job成功。Ubuntu / Windows × Python 3.13 / 3.14各123件、実モデルPython照合M1 15件 + 検索10件が合格。macOS / iOS / AndroidとPlayerでの実モデル実行は未検証。
- mainはbranch protection設定済み。PRとRequired CIを管理者にも要求し、force push / 削除を禁止。全PRのCI入口とパッケージ監査はmainへ統合済み。クラウドUnityはLinux CPUの任意手動補助検証で未実行、2026-10-10確認時にSecrets / Variablesは未登録。

## 作業順序と完了条件

| 順序 | タスク | 完了条件 | 依存 / 状態 |
| --- | --- | --- | --- |
| 0 | OSS開発基盤: CIの必須判定とmain保護 | 文書だけのPRも含めCI判定が完了する。失敗・必要jobの未実行を成功扱いにしない。PR経由、必須check、force push / 削除の制限をGitHub側で確認 | 完了。保護を再確認、PR #6 / #8統合済み、統合後mainの全8 job成功 |
| 1 | UPM構成へ移行 | package manifest、明示的な依存、Runtime / Tests / Samples / 文書を整理。既存API契約とGUIDを維持し、重複assemblyやUnityEditor参照を持ち込まない | 完了。PR #6統合済み、新規consumerのローカルフォルダ導入・契約29件合格。Git URL検証と公開は6で扱う |
| 2 | テキスト検索サンプル | 文書を事前埋め込みし、queryとのcosineで順位表示。固定入力のPython参照順位と一致。モデル未準備・不正入力・実行失敗を表示し、終了時にリソース解放 | 検索基盤main統合済み。PR #11でサンプル実装、Windows Sentis4条件と元プロジェクト画面操作を確認。PR統合が残る |
| 3 | モデル準備・配布手順 | CIで固定revisionから生成、hash監査、取得・配置・読み込み・更新の手順を整備。新規checkoutとサンプル導入で再現 | consumer内の導入・監査・変換と数値照合を保存。高速化・CLI完了応答・consumer画面操作の確認が残る |
| 4 | 環境・runner・実機の確保 | macOS Editor、iOS / Androidのtoolchain・実機・GPU API・ライセンス・署名条件を確認し、実行できる組み合わせを記録 | 早期に調査。必要環境は未確認 |
| 5 | 各環境の実モデル検証と測定 | 下の環境表を埋め、精度・実行backend・ロード / 初回 / 定常時間・メモリを記録。移行後Windowsの回帰も確認 | 移行後Windowsの導入・契約・M1 / API回帰済み。保存・量子化・全ベンチ再測定、サンプル、他環境は残る。1〜4に依存 |
| 6 | リリース準備と配布 | clean projectから版固定で導入、サンプル起動、手順・ライセンス・CHANGELOGを確認。必要環境の合格後にリリース用PRと配布を行う | 1〜5に依存、未着手 |

0 / 1は完了した。直近は不要なimport / 再変換 / Refreshと外部キャッシュ待機の削減 → consumer再現の完了応答・画面操作 → 検証記録・手順の確定 → PR #11統合・統合後CI。続いて4の環境確保 → 5の実測 → 6のリリースへ進む。
今後も作業ブランチからPRを提出し、mergeは依頼された範囲でCI・差分・競合を確認して行う。

## 直近のPR分割と受け渡し

| 順番 | PRの範囲 | 次の作業へ渡すもの / 合格条件 |
| --- | --- | --- |
| A | 検索の固定ケースと順位計算 | 日本語 / 英語の小さい文書群・query・同点順序を定義。Python参照生成とC# cosine / ranking契約をTDDで追加し、Actionsで固定revisionの期待順位を生成。小さいfixtureの合格と実モデル照合を別記 |
| B | `Samples~/TextSearch/` の操作とモデル読み込み | Aの順位計算と既存TextEmbedderを使用。文書の事前埋め込み、query入力、順位表示、未準備・不正入力・実行失敗、終了時DisposeをTDDで実装。Windows Editorでfp32 / Float16重み、CPU / GPUComputeを参照順位と照合し、UI操作を記録 |
| C | 利用者向けモデル準備と再現手順 | 既存prepare / stage CLIとActionsを使用。revision・source SHA・hash、取得・変換・配置・更新、artifact失効時の再生成を文書化。新規checkoutからBのサンプルを起動して再現。重みはGitへ含めない |
| D | 他環境の実行と配布準備 | 下表のtoolchain / runner / 実機を確保し、実モデルの精度・backend・速度・メモリを記録。Git URLのcommit固定導入を実Editorで確認し、必要環境の合格後にtag固定導入とリリースPRへ進む |

A → Bの順に進め、Cの手順整理とDの環境調査は早期に進められる。Cのサンプル再現はBに依存する。
実装ではテストを追加 → 意図した失敗を実行確認 → 最小実装 → greenを記録する。Pythonはtools/とuv、重い参照生成・変換・全テストはActionsへ寄せる。
Unityや実機が利用できない環境は未実行として残し、サンプルや公開の完了条件を満たしたことにはしない。

## UPM構成

PR #6で `com.ayutaz.embeddinggemma`、開発版 `0.1.0-pre.1`として実装。公開tagはまだない。
[Unity公式のパッケージ構成](https://docs.unity3d.com/6000.3/Documentation/Manual/cus-layout.html)に沿って整理する。

```text
Packages/com.ayutaz.embeddinggemma/
  package.json
  Runtime/
  Tests/Editor/
  Samples~/TextSearch/  # PR #11で導入可能。モデル重みは別途準備
  Documentation~/
  README.md
  CHANGELOG.md
  LICENSE.md
```

- Runtimeの `.meta` / GUIDとassembly名を維持して移動し、元のAssets側に同じ実装を残さない。
- package manifestではSentis 2.6.1と直接使用するNewtonsoftの依存を明示する。新規プロジェクトで依存解決を検証し、現行lockの偶然の解決に頼らない。
- モデル不要の契約テストをパッケージ側へ整理。既存の `Assets/M1Generated/` や `artifacts/` を使う実モデルfixture・計測runnerは、検証プロジェクト側に残すかパスを注入できる形に分離する。
- 消費側はローカルフォルダ依存で別Editor導入・29件合格を確認済み。[Git URLのsubfolder / revision指定](https://docs.unity3d.com/6000.3/Documentation/Manual/upm-git.html)はmanifest生成・監査のみで、Editorでの導入は未実行。commit固定で検証し、リリース時にtag指定の手順を用意する。
- まずUnity 6000.3.16f1を検証対象とする。下位のUnity / Sentisへの対応は別検証なしに宣言しない。

## サンプルとAPIの確認

サンプルは日本語 / 英語の小さい固定文書群を用意し、`EmbedDocument` で埋め込み、`EmbedQuery` の結果との全件比較で順位を表示する。
固定queryと期待順位はモデルrevision・prompt・tokenizer・精度条件を揃えたPython参照からCIで生成する。
同点時の順序を固定し、fp32 / Float16重みそれぞれで参照順位と照合する。任意の利用者入力に対する検索品質を固定テストの成功から保証しない。

順位計算、不正入力、Dispose、未準備モデルの処理は先に失敗テストを確認して実装する。
サンプルはモデルを自動取得せず、利用者が準備したモデルとtokenizerを指定する。毎回の検索で保存・量子化しない。
UI操作で文書準備 → query入力 → 結果表示 → 終了まで確認し、スクリーンショットと小さい結果要約を残す。
非同期化・可変長・batch拡張は現行APIの必須条件ではなく、必要性と別の契約を定めてから扱う。

## 環境別の受け入れ

| 環境 | 現状 | M2で必要な確認 |
| --- | --- | --- |
| Windows Editor | M1完了。UPM移行後の別consumer導入・契約29件、元プロジェクトのM1 / API回帰済み。検索サンプルはPR側で保存済み2形式 × CPU / GPUComputeの全順位合格 | サンプル画面操作・新規consumer実モデル再現、Git URL導入、全測定の再確認 |
| macOS Editor | 未実行 | 使用機種 / OS / GPU APIを記録し、実モデルCPU / GPUComputeの精度と測定 |
| iOS Player | 未実行、toolchain / 実機未確認 | IL2CPP・stripping・モデル配置を含むbuild、実機CPU / GPUCompute実行、精度と測定 |
| Android Player | 未実行、toolchain / 実機未確認 | ABI / graphics API・モデル配置を含むbuild、実機CPU / GPUCompute実行、精度と測定 |
| Linux CPUクラウド | 任意手動workflowのみ、未実行 | Secrets準備後に補助検証。他の必須環境やGPUの合格と区別 |
| Web / WebGPU | 任意、未着手 | 必須環境の後に対応可能性を調査 |

精度はM1と同じ15件のtoken ID / mask完全一致、有限・768次元・単位長を維持する。
fp32はPython参照cosine >= 0.999、Float16重みは >= 0.99。GPUのskipやCPUへの切り替えでGPU合格にしない。
対象端末で未対応・OOM・build失敗となった場合は、実行条件と失敗を残して対策を検討し、対象環境を黙って完了扱いにしない。

PlayerではEditorのimport処理と分け、事前変換済み `.sentis` とtokenizerを配置して起動時に読み込む。
読み取り専用の配置先と書き込み可能な保存先を分け、`File.Replace` を含む保存APIの対応を各対象環境で確認する。
ロード・初回・warmup後の推論を分けて測る。メモリは測定API・取得可否・観測範囲を記録し、Editor全体の値からモバイルの必要量を推定しない。
Float16重みは保存サイズ削減済みだが、M1では速度改善は確認されていない。端末の採用形式は実測から判断する。

## CIと証拠

- Python・参照生成・変換・パッケージ静的検証・可能なUnity build / testsはActionsを優先する。Python変更はtools/とuvでTDD。
- 必須CIは全PRで完了する入口を用意し、変更範囲に応じたjob結果を集約する。対象外と必要jobの失敗 / cancelを区別する。
- クラウドUnity実行にはライセンス設定が必要。GPUやモバイル実機のrunnerがあるとは仮定せず、可能なCIと必要最小限のローカル / 実機実行を分ける。
- Secretsを使う実行と外部forkのコードを分離する。未実行のクラウドUnityを成功として記録しない。
- run URL、PR head、実際のcheckout SHA、固定モデルrevision、artifact hash、実行環境、backend、数値、失敗・skipを保存する。
- モデル・大きいログはGit管理外、LFSは使わない。artifact失効時はCI再生成し、小さい数値要約と再現手順をGitに残す。

既存のworkflow構成と制約は [CI手順](ci.md)、ローカルの必要時操作は [uloopハーネス](automation.md)。

## M2の後

M3で画像エンコーダとGPU上の画像前処理、共通テキスト本体への特徴量入力、テキスト→画像の固定参照順位照合を実装する。
M4で音声エンコーダとGPU上のメルスペクトログラム前処理、共通空間への入力、テキスト→音声の固定参照順位照合を実装する。
どちらも実装未着手。演算子互換性・中間テンソル・メモリ・CPU readbackの有無を調査してから詳細計画を作る。
全体の完了条件は [ゴール](goal.md)を維持する。
