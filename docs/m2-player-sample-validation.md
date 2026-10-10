# Windows sample Player の検証

2026-10-10。配布 TextSearch sample を実際の Windows Player に含め、GUI 表示・検索・解放を確認するためのハーネス。
既存の M1 全条件 Player とは別の起動 marker `EmbeddingGemmaSampleValidationBuild` を使う。
marker がない通常の Player と、Editor の Play では自動実行しない。

## 実装と TDD

- 新規の sample proxy / 全6件 ranking / build 設定契約: 12件中11 failed → 全12 passed。
- 実行と失敗時解放、結果保存、linker、scene / corpus の provenance: 新規12 failed → 全24 passed（上の12件を含む）。
- Player 起動条件とモデル欠落: 5件中2 failed → 新規全29 passed。failed / skipped / inconclusive は0。
- GUI の遅延準備: 新規1 failed → 全31 passed。GUI preflight: 新規1 failed → 全32 passed。モデルを準備する前に最大120 render frame待ち、GUI 不成立ならモデル監査・準備を行わない。
- Player validation assembly のまとめ実行は139 passed / 2 failed / skip 0。失敗した2件は Android target が必要な build integration 契約で、この Editor は StandaloneWindows64。対象違いの失敗を残し、Android 成功とは数えない。

初期 source は `239992584b93142d7563be0156e95f7e1ffe962b`、描画待ち・診断は `904879d9cfa304d8fcc4210d45f52c45083facbc`、モデル前 preflight は `747f986c159d6ba78490c2b207e98c1ceb86af02`。
ビルドには `BuildWindowsSample` を使い、import 済み `Assets/EmbeddingGemmaTextSearch/TextSearch.unity` を選ぶ。
IL2CPP / compiler Release / High stripping / BuildOptions.None を指定し、元の scene と backend / stripping / compiler 設定を成功・失敗時とも復元する。
注入した build を実 Player build 成功にしない。

`Assets/Validation/Runtime/link.xml` は検証 assembly と optional sample assembly を保持し、Core の Worker 所有者の field を保持する。
これは validation 用設定で、配布 UPM 本体には追加しない。任意 consumer の stripping 成功とは扱わない。
optional assembly の `ignoreIfMissing` と fields の保持形式は [Unity 6.3 の link.xml reference](https://docs.unity3d.com/6000.3/Documentation/Manual/managed-code-stripping-xml-formatting.html) に基づく。

## 実行の範囲

ハーネスは sample の公開 API を呼ぶ。native keyboard / mouse 操作の検証ではない。
Float16 重み / actual GPUCompute Worker / 固定6文書 / 4 query の全順位と有限 score（誤差上限0.02）を確認する。
先頭の日本語・英語 query ごとに、Player 自身が end-of-frame 後に1280×800の PNG を保存する。
空入力の拒否、Worker と cache lease の解放、モデル欠落の拒否、view がある間の native font と view 破棄後の font 解放も確認する。
模擬 view / loader / capture を注入したテストは `injected_contract` となり、GPU 実行合格にはしない。

同じ binary の独立2起動で、結果ディレクトリ内の専用 `sample-model-cache` を共有する。
既存モデルは外部 bundle から `file://` を使い、sample の cache に配置する。全5ファイルを固定 Python 参照付き bundle loader で完全監査し、sample cache でも選択した model / tokenizer の完全 hash を照合する。
同じ preparation receipt と bundle の source / 元model hash / 全3モデル descriptor の一致を起動前にも確認する。
既存の準備済みモデルと Editor import / shader cache は再利用する。新規 download / conversion、cold-machine 性能の証拠ではない。

## 実行状況

[source・TDD・build・失敗実行の証拠](results/m2-player-sample-windows-20261010.json)を保存した。最新ビルドの37ファイルの正規化済み source hash が固定作業ブランチと一致した。
consumer は Git SHA `fc66af7` を解決した既存の `artifacts/j` を利用し、sample の測定 fixture 2ファイルと validation source を作業ソースで配置した条件。新しいハーネス SHA の空 Git 導入成功とは扱わない。

| source | 実 build 秒 | errors / warnings | Player 観察 |
| --- | ---: | --- | --- |
| `2399925` | 231.830 | 0 / 485 | Direct3D12。実 Worker は GPUCompute、cold cache は3ファイル転送・完全CNG hash。GUI font 不在で query 前に失敗 |
| `904879d` | 41.763 | 0 / 485 | Direct3D12。120 frame待っても font 不在。Direct3D11 の同 binary 比較も同じ。後者の cache はmanifestだけ1転送・完全CNG監査・再利用だが、query は0 |
| `747f986` | 52.457 | 0 / 485 | Direct3D12。GUI preflight が失敗したためモデル監査・準備を未実行。カメラと viewport の状態を保存 |

初回 build と差分 build は異なる source で、全て既存 cache 利用。41.8秒 / 52.5秒は同一条件の統計的高速化率ではない。
各 build は受理後の同じ operation を観察し、再投入・Editor 追加起動なし。StreamingAssets の既存モデルは退避後に復元済み。
元の sample scene（clean）、Play 停止、StandaloneWindows64、Mono2x / Disabled / Release への設定復元を確認した。

5つの Player process は全て `success=false` / `gpu_verified=false`、終了 code `-1073741819`（`0xc0000005`）で異常終了した。
最初の crash dump の例外アドレスは `UnityPlayer.dll+0xb12f91`。記録した候補アドレスは symbol 解決・stack unwind をしておらず、原因を特定した証拠ではない。dump と大きい binary / モデルは Git に含めない。
最新版でも推論器を作る前に同じ異常終了が再現したため、Sentis Worker の生成・解放はこのモデル不要の再現には必要でない。GPU graphics device 自体は使用しているので、GPU/driver を全面的に除外しない。

Direct3D11 / 12 の両方で1280×800の診断画像は全面黒、同じ完全 SHA-256 `bbdccdfa0f0eb1f22460f9faa6cd64dec41db193a88d8de9453e469e62e8fb6e`だった。
sample は active / enabled / useGUILayout=true、splash 完了済み / batch=false / focused=true。
最新観察は frame 122、active camera 1、Main Camera active / enabled、viewport 全面、1280×800、Forward、背景 RGBA(0.060,0.080,0.120,1)、target texture なし。Editor の同じ項目と一致した。
同じ `747f986` binary を [Unity 6.3 公式の `-force-gfx-direct`](https://docs.unity3d.com/6000.3/Documentation/Manual/PlayerCommandLineArguments.html) でも比較し、実ログの `kGfxThreadingModeDirect` を確認した。font不在・モデル未実行は同じで、診断PNGの texture取得も失敗した。単一スレッドでも終了時アクセス違反が再現したため、これを修正とは扱わない。
binary の Resources には sample 用 marker を確認し、既存の全条件検証 marker は検索結果になかった。CI の成功や marker 分離契約だけで実描画の原因を特定しない。
これは可視 UI が表示できた証拠ではない。query 検索、空入力 / モデル欠落 / native font寿命、独立2回の sample 合格は未達。

![失敗時の実Player診断画像。全面黒でsample UIは未表示](images/m2-player-sample-gui-failure-windows.png)

`Start-Process -WindowStyle Hidden` で開始した条件を保存した。内部の focused=true だけで OS window の可視状態を確定しない。
表示条件の比較はユーザーの希望確認待ち。今後はモデルなしの描画 / shutdown を先に切り分け、同じ binary の表示条件で比較する。
原因・修正確認が済むまで [PR #31](https://github.com/ayutaz/unity-embeddinggemma-2/pull/31) は draft とし、CI 成功を実Player成功の代用にしない。

## モデル不要の終了処理の比較

以前の精度検証で合格した `c742fb940b78393573b19e42352fe7b70256e74d` の IL2CPP / Release / High stripping binary を再利用した。元の実モデル2起動も保存済み arguments の `-batchmode -force-d3d12` を確認した。sample scene と新しい sample ハーネスを含まない既存 Player に、存在しない bundle directory を指定した。全条件で `DirectoryNotFoundException` を結果に保存し、model / tokenizer / Worker / query は実行していない。

[3起動の比較条件・終了結果・元 report / log の hash](results/m2-player-shutdown-comparison-windows-20261010.json)を保持した。公開用の要約にはユーザー名・絶対パス・PID・機種情報・生の crash record を含めず、完全な原本は Git 管理外の local artifacts に維持した。前の process の終了を確認してから次を起動し、全て `Start-Process -WindowStyle Hidden`。新規 build・モデルの download / copy・Editor の追加起動はしていない。

| 条件 | 実際の graphics API | process 終了 code | 観察 |
| --- | --- | ---: | --- |
| 通常モード、非表示、1280×800 windowed | Direct3D12 | -1073741819 (`0xc0000005`) | 失敗報告保存後にアクセス違反 |
| `-batchmode -nographics` | Null | 1 | 想定した欠落エラーの終了 code で停止 |
| `-batchmode -force-d3d12` | Direct3D12 | 1 | 想定した欠落エラーの終了 code で停止 |

通常モードの旧 Player dump も sample Player と同じ native 例外位置だった。旧 Player と sample Player の `UnityPlayer.dll` は全ファイル SHA-256 `bef601ad70832ed45da76262346d90bd8136d97f2d338db743f02d9b2f314712` が一致した。これは共通の例外位置を示すが、symbol 解決 / stack unwind を行っておらず、原因特定ではない。生の例外情報は公開要約に含めない。

sample font / scene / 新ハーネスと、モデルの準備は、旧 Player のこの再現には必要でない。batch mode で graphics device が有効でも終了できたため、graphics device 初期化だけを原因と断定しない。batch mode は window と実行条件も変えるため、単独要因の同定には通常モードの可視表示との比較が残る。以前の実モデル合格は当時の実行条件の証拠として維持し、この新しい欠落条件へ一般化しない。

`-nographics` の意味は [Unity 6.3 公式 Player command-line reference](https://docs.unity3d.com/6000.3/Documentation/Manual/PlayerCommandLineArguments.html) に従う。batch mode の想定どおりの失敗終了は GUI / GPU 推論の合格ではなく、全3報告の `success=false` / `gpuVerified=false` を維持する。非表示起動を GUI 検証の推奨条件とは扱わず、ユーザーの表示希望を確認してから同一 sample binary の通常表示を比較する。

## 再現

Unity 6000.3.16f1 / Sentis 2.6.1 の固定 Git consumer に TextSearch sample を import し、validation 用 `Assets/Validation` を固定 source から配置する。
scene を保存し、Play と Test Runner を停止する。モデルを Player に同梱しない場合は既存の StreamingAssets のモデルを owned staging へ退避し、build 完了後に戻す。

```csharp
ValidationPlayerBuild.BuildWindowsSample("<new empty build directory>/Sample.exe", "<fixed 40-character source commit>");
```

次は今回失敗した非表示条件の再現であり、GUI の受け入れ確認に推奨する起動条件ではない。通常表示の比較結果は未取得。

```powershell
$arguments = @('-screen-width','1280','-screen-height','800','-screen-fullscreen','0','-force-d3d12',
    '-logFile','<new owned results directory>/cold.log',
    '--embeddinggemma-bundle','<audited local bundle including preparation.json>',
    '--embeddinggemma-output','<new owned results directory>/cold.json',
    '--embeddinggemma-run-id','<new 32-character lowercase hex id>')
$player = Start-Process '<build directory>/Sample.exe' -ArgumentList $arguments -WindowStyle Hidden -PassThru
$player.WaitForExit()
```

最初の process の終了 code / report / PNG を検査してから、同じ results directory に新しい run ID・`warm.json`・`warm.log` を指定して2回目を起動する。
既存結果先の上書き、受理済み operation の timeout による再起動はしない。

macOS / iOS / Android 実機、Editor 検索 DB と native allocation の診断、正式候補 / tag の導入と Release は引き続き別の未達 gate。
