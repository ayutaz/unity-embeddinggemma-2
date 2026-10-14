# GPU constantのIL2CPP保持・実行検証

> 文書整合確認: 2026-10-14。[現在の状態](status.md) / [GPUを主経路とする再開計画](multimodal-tdd-plan.md)。本文の数値・source・実行中／次の作業は記載当時の記録。現在headの実行結果として扱わない。

2026-10-14 JST。[Draft PR #71](https://github.com/ayutaz/unity-embeddinggemma-2/pull/71)。baseはGPU分割転送PR #69。最新の実PlayerはGUID修正済みの固定Git `adccab24dbcc3f5dd8c8ac79911a7604a6014021`を使用する。初回`f246eea58543f806aca1d880c2c37d0eee86fd21`の実行も保持する。[実行receipt](results/gpu-constant-il2cpp-windows-20261014.json)。正式Release / tagは作成しない。

## パッケージの保持設定

Unity 6.3は、Assets配下の`link.xml`を自動発見するが、パッケージ内の同名fileを自動発見しない。[Unity公式仕様](https://docs.unity3d.com/6000.3/Documentation/Manual/managed-code-stripping-preserving.html)。RuntimeのXMLを置くだけでは、明示的な保持設定を供給したことにならない。

Editor専用assemblyの`GpuConstantLinkerPreservation : IUnityLinkerProcessor`から、解決済みパッケージの`Runtime/link.xml`を`GenerateAdditionalLinkXmlFile`へ返す。Workerの`m_Storage` / `m_LayerCPUFallback`、ModelStorageの`GetTensor`だけを明示する。パッケージが未解決、またはXMLが欠ける場合はbuild failureとする。Runtime assemblyへUnityEditor依存を追加しない。

callback不足を1 assertion redで確認し、固定Git consumerで1 green / skip0 / CLI exit0。package全254 green / failed0 / skip0 / CLI exit0。要求・解決Gitは同じ`f246eea…`で、Runtime source overlayはない。実ビルドのUnityLinker responseには、このパッケージのXMLが`--include-link-xml`として含まれた。responseのSHAをreceiptに記録する。

## TDD

| 対象 | red | green |
| --- | --- | --- |
| 実行可能なCPU / GPU probe | API不足2 assertion、skip0 | 2 passed、skip0 |
| IL2CPP High / Release / D3D12 buildと状態復元 | builder不足2 assertion、skip0 | 成功注入 / 例外注入の2 passed、skip0。注入buildのsuccessはfalse |
| Player結果判定 | runner不足2 assertion、skip0 | 2 passed、skip0。GPU不一致を拒否し、注入結果を実Player成功にしない |
| package linker callback | callback不足1 assertion、skip0 | 1 passed、skip0。保持対象とpackage解決pathを確認 |

probeの6件をまとめた実行も6 passed / failed0 / skip0。各CLI結果の完了時刻・SHAをreceiptへ保持する。builder / runner / probeの初期TDDはRuntime `66c72fb…`、リンカー修正・package回帰は`f246eea…`の結果であり、現在headへの再実行に付け替えない。

## 実Windows Player

Unity 6000.3.21f1 / Sentis 2.6.1、Windows64、IL2CPP、High stripping、Release compiler、BuildOptions.None、Direct3D12。既存consumer `artifacts/k`を、Editor1個でuloopからbuildした。専用Resources markerを使い、旧検証Playerの起動markerを生成しない。build終了後に一時scene / markerを削除し、設定とsceneを復元する。成功・例外時の復元は上記TDDで検証した。

初回buildはSucceeded、214.3889681秒、613969078 bytes、error0 / warning185。GUID修正後の増分buildもSucceeded、36.4871121秒、同bytes、error0 / warning185。増分時間を一般的な性能改善倍率にはしない。初回185件の警告はBuildReport APIでも回収し、未対応Sentis convolution variant160件、integer modulus7件 / divide2件、signed / unsigned15件、loop1件へ分類した。warning0とはせず、probe外の全shader / variantの動作を判定しない。

約1.7 GBの既存StreamingAssetsモデルはprobeには不要なため、Assets外へ一時退避し、2起動の終了後に元のconsumerへ戻した。モデルの削除・再download、重いgraphの変換や再保存は行わない。binary、shader cache、モデル、大きなログをGitへ含めず、build / executable / GameAssembly / resultのSHAだけを記録する。

GUID修正後の独立2プロセス（PID 104696 / 163844）はともにexit0。初回sourceのPID 69608 / 139772もexit0だった。各回でCPU / GPUComputeの2条件、全2097153 float要素のbit mismatch0。GPUは1 constant / 2 SetData calls / 最大8388608 bytes、CPUはupload0。holder破棄後の借用Model再利用も合格。`is_editor=false`、`il2cpp=true`、`development_build=false`、実graphics APIはDirect3D12、GPUはRTX4070 Ti SUPER。重みをdownloadしない独立graphのprobeであり、実512 MiB constantや実画像 / 音声chainのPlayer受け入れとは別。

埋め込んだbuild markerはbuild前のprovenance snapshotであり、その`success=false`はbuild結果ではない。実buildのSucceeded / success=trueは別の`build.json`、実行成功は各runの`success=true`とprocess exit0へ記録した。

## CI失敗・未実行

初回head `f246eea…`の[CI 38443470547](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38443470547)で、Editor asmdef metaの33桁GUIDがpackage audit / Python testsを不合格にした。`adccab2`で32桁へ修正し、[CI 38443982321](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38443982321)はmodel-reference / Required CIを含め全8項目成功（18:18:29 UTC完了）。元CIの失敗を消さない。GUID修正headは実buildと2起動を追加実行し、初回sourceのPlayer結果を付け替えず両方を保持した。後続の文書commitのCIは別runとして追跡する。

GUID修正後のconsumerはrequested / resolved Git `adccab2…`が一致し、Runtime overlayなし。compile error / warning0、linker test1 passed / failed0 / skip0 / CLI exit0（18:12:36 UTC完了）も別receiptとして保持する。

GitHub Unity secret name一覧は0件のため、ActionsでのUnity build / 実行は未実行。今回だけ必要な小さいPlayerをローカルでbuildした。

未完了: 元の画像 / 音声encoder数値gate、実モデル全chainのIL2CPP、Raw / Query・入力境界、長い反復 / memory profiler、Byte専用case、sample、macOS / iOS / Android実機、最終空consumer、配布準備。工程1〜6の完了や正式リリースの合格にはしない。
