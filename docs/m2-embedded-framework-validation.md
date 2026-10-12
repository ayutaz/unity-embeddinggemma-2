# Unity21の空Git consumerとembedded Test Framework検証

確認日: 2026-10-13。M2 / C1の導入再現を進める。素のUnity21同梱Test Framework 1.6.0には2つのC# `.meta`欠落があり、immutableなpackageでのcompile失敗は過去の記録として保持する。今回は同じ公式sourceをconsumerのembedded packageへコピーし、Unity自身が不足metadataを生成する手順で検証した。**未変更の標準環境が合格したという結果ではない**。

## 手順とTDD

`tools/embeddinggemma_tools/package.py`に任意の`--test-framework-source`を追加する。新しい空consumerだけが対象。package名 / 固定version / licenseを確認し、元ファイル全件のSHA-256・サイズと欠落metadataをreceiptへ記録してから、`Packages/com.unity.test-framework`へコピーする。C# sourceやEditor本体・PackageCacheを手修正しない。Unity公式の[embedded package](https://docs.unity3d.com/6000.3/Documentation/Manual/upm-embed.html)はmutableで、同名依存より優先される。

```powershell
cd tools
uv run --locked python -m embeddinggemma_tools.package --consumer ../artifacts/new-consumer `
  --git-revision fde79e2ab11210e711db14acb6ccc8c758a09229 --automation --sample `
  --test-framework-source '<Unity21 Editor>/Data/Resources/PackageManager/BuiltInPackages/com.unity.test-framework'
```

- helperのコピー・source拒否・既存consumer保護5件の未実装redを確認。CLIでoptionを無視しない契約も1 red → green。関連package / Git解決の最終回帰46 passed、skip 0。
- receiptの`unity_compiled=false`はコピー時点の証拠。これをcompile成功と読み替えず、以下の実Unity結果を別に記録する。
- fake sourceの単体テストはhelperの契約だけを確認する。実Editorの検証には署名を確認済みの6000.3.21f1公式sourceを使った。vendor sourceはGit / UPMへ追加しない。

## fresh consumerの実測

以前のLibraryがない`artifacts/k`を作成し、Unity 6000.3.21f1 / Sentis 2.6.1 / uloop-cli 3.8.1で実行した。Editor本体は1つ。初回import中の一時的なbusy応答では再起動せず、同じEditorと保存結果を追跡した。

| 対象 | 結果・範囲 |
| --- | --- |
| Git導入 | requested / resolvedとも`fde79e2ab11210e711db14acb6ccc8c758a09229`、sourceはgit。Runtime / sample codeのoverlayなし |
| framework | source 1,168ファイル全件のhash不一致0。Unityが不足する2つの`.cs.meta`を生成。公式builtin sourceは変更しない |
| compile | Success、Error 0 / Warning 0 |
| UPM EditMode | 38 passed / failed 0 / skipped 0 |
| sample EditMode | 67 passed / failed 0 / skipped 0。CLIはaccept後に切断したため、その実行が保存したNUnit XMLの開始 / 終了時刻・assembly・件数から回収。CLI通信失敗も保持し、テストを重複実行しなかった |
| sample PlayMode | 2 passed / failed 0 / skipped 0、CLI完了応答を取得 |
| 実モデル検索 | FP32 / Float16 × CPU / GPUComputeの4 passed、failed / skipped 0。固定6文書 / 4queryの全6順位とscoreを参照照合 |

検索のminimum cosineはFP32 CPU `0.9999999999981735`、FP32 GPU `0.999999999999512`、Float16 CPU `0.9999999999982625`、Float16 GPU `0.9999999999994984`。NVIDIA GeForce RTX 4070 Ti SUPER / Direct3D12。数値は[保存結果](results/m2-embedded-consumer-windows-20261013.json)を参照する。

実モデルは監査済み既存artifactを再利用。モデルsource `2edb272f17a38c7af210474345c35f391fd6955c`、検索参照source `13f194266162d2a728b2b2a0da9e9e58ae314ff6`、model revision `914f7f89142e33e77833254d9c9b90c3cef7303b`を維持する。参照照合用Editor fixtureのみconsumerのAssetsへ配置した。packageの実装を差し替えた結果ではない。

## 未達と制約

- embedded workaroundを使った導入成功であり、builtin package欠落自体の修正ではない。
- 新consumerの日英UI・欠落モデル・cache操作、token15件、独立終了、新candidateのnative Player / 実入力はまだこの実行の証拠に含まれない。既存の旧source実測は別証拠として保持する。
- raw Editor logのD3D12 upload buffer拡張診断（16,777,216 → 要求536,870,912 bytes）を保持する。性能 / memory改善やdomain reload・allocation問題の解決は主張しない。
- macOS / iOS / Android実機、M3 / M4は未検証。PR #31やM2全体を自動的に完了としない。正式リリースは行わない。

大きいmodel・vendor source・生XML / logはignored artifactsへ置き、Gitには小さい結果要約だけを保存する。
