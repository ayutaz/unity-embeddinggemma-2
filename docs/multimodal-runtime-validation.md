# 共通マルチモーダル本体のRuntime契約

更新日: 2026-10-13。[全体TDD計画](multimodal-tdd-plan.md)のC3 / C4を進める。`MultimodalBackboneProfile`と`MultimodalBackbone`をUPM Runtimeへ追加する。**画像 / 音声encoder、公式prompt構築、GPU前処理、実モデルの最終768次元照合はこの変更の完了範囲に含めない。**

## APIと所有権

- 固定batch 1、長さ2〜8192、image / audio feature数とplaceholder ID、text vocabulary / pad IDをprofileで指定する。feature幅512、最終embedding幅768は固定。
- int32 IDs / binary maskの長さ、非空mask、通常IDの範囲、masked placeholder、image / audio slot数を推論前に拒否する。feature順の数値処理は保存graphが担当し、Runtimeはtoken順とfeature row順を変更しない。
- Modelはstaticな4入力 `input_ids` / `attention_mask` / `image_features` / `audio_features`と1出力を要求する。名前、型、shapeをprofileへ照合する。
- 呼び出し側がModelとproject済みfeature Tensorを所有する。backboneはWorkerと各呼び出しのIDs / maskのみを所有し、borrowed featureをDispose / CPU downloadしない。
- 非空featureは既に指定backendへ置かれたlive Tensorでなければならない。CPU / GPUComputeを明示し、GPUPixelや非対応端末を拒否する。GPUへ自動uploadするfallback APIにはしない。
- 最終outputのshape / actual backend、全finite、L2 normを検査する。最終vectorだけを独立したCPU配列へコピーし、Disposeは冪等。Dispose後の推論は拒否する。

GPU featureの全値をhost検査するためのreadbackは行わない。呼び出し側でencoder / artifactのfinite契約を検証し、Runtimeは最終vectorのfinite条件も確認する。画像 / 音声なしのzero-feature profileは型として許容するが、実profileのimport・zero-size tensorを含む実GPU受け入れは別途必要。

## TDDと実行範囲

基準mainは `e35470212f616b020c0b150a4c515a007f0b937f`。未コミットの未実装stubと先に書いたテストを、既存consumer `artifacts/k`のAssets内test assemblyへ配置して実行した。Runtime package / vendor cacheを上書きしていない。配布assemblyへの新規Git導入の証拠とは区別する。

- compile成功後、初回は22件中21 failed / 1 passed / skip 0。profile / layout / model契約、不正output、GPUでのCPU feature拒否、所有権・結果の数値について意図したassertion失敗を確認。
- 最小実装後は22 passed / skip 0。
- 小さい実GPUの追加受け入れを含め23 passed / skip 0。RTX 4070 Ti SUPER / Direct3D12、Unity 6000.3.21f1 / Sentis 2.6.1。GPU featureのbackend維持、実worker / output、CPU layer計算とreadback flushなし、Dispose後もcallerのTensorがliveであることを確認した。小さいgraphの結果であり、実model / 1024長の合格ではない。
- 破棄featureを渡す経路で、APIのArgumentExceptionではなくSentis内部assertionが発生する追加1 redを確認。dataの生存をbackend参照より先に検査する最小修正を追加した。
- 最終は新共通API 24件と既存UPM契約38件の計62 passed / skip 0。compile成功、上記の実GPU受け入れを含め合格。[source hashと段階別結果](results/multimodal-runtime-windows-20261013.json)。
- `tools/`からuvで配布監査・別consumer manifest準備が成功。新source / meta / GUID・Runtime依存とモデル混入を監査した。このmanifest準備はUnity起動・Git解決の証拠ではない。

元のCLI呼び出しに存在しない`--timeout`を渡した引数エラーは、テストが未実行であるためTDD redへ含めない。CLIが案内した`--timeout-seconds`で実行した。Editor本体は既存PID 78588の1つを再利用し、重いmodel実行・新規Editor起動は行っていない。

## 次の受け入れ

1. 上記62件の契約greenを起点に、small graph / 実GPU / 実modelの結果を引き続き分離して保存する。
2. 監査済み実backbone exportの1024長・実encoder feature数profileをimport / 保存・再読込し、このAPIで独立Pythonの最終768次元へ照合する。CPU / GPUCompute × FP32 / Float16を別判定する。
3. 公式processorのprompt / 特殊token・maskを生成するAPI、ImageEmbedder / AudioEmbedderとGPU前処理をTDDで接続する。text128長の既存APIは維持する。
4. 固定candidateの空Git consumerへRuntime assemblyを導入し、sample・Player・全必須実機へ進む。正式Release / tagは作成しない。
