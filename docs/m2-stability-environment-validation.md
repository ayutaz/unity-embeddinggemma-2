# Windows Editorの起動環境とモデルなし安定性の切り分け

確認日: 2026-10-10。main `d6c84424498e5bfb67c0adf3b1292a66e97c3a7c`はPR #26統合後。[PR head CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38027163949)と[main CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38027319740)の全8 job成功を確認した。
[機械可読結果](results/m2-stability-environment-windows-20261010.json)を参照。Android実APKのbuild source `751c30a`と[実APK結果](m2-android-validation.md)は変更しない。

## 起動環境

回復時にuloopを直接起動したEditor PID 127008では、`ALLUSERSPROFILE`が欠けていた。`ProgramData`とCommonApplicationDataは`C:\ProgramData`だった。モデルなし空sceneのPlay / Stopで、Package Manager Windowの`The "path" argument must be of type string. Received undefined` Errorを1件観測した。これは実APK build成功とは別の失敗として記録する。

元scene・設定・sampleデータは復元済み、保存済み・Play停止・compile停止を確認した。通常終了を予約したがPIDは残りReadyだったため、同じ保存済み検証Editorだけを終了した。旧PID終了・Editor本体0を確認してから、既存のuv Pythonハーネスを使って1回だけ起動した。タイムアウトを終了証拠にはしない。補助workerの1つは親終了時に自然終了し、追加Editorを起動する前の本体0確認を維持した。

`tools/embeddinggemma_tools/unity.py`はWindowsで`ALLUSERSPROFILE`が未定義なら`ProgramData`を補完する実装を既に持つ。この実装は変更していない。toolsでuvを使い、対象consumerとCLIを明示する。

```powershell
uv run --locked python -m embeddinggemma_tools.unity `
  --project <consumerの絶対パス> --uloop <uloop.exeの絶対パス> `
  --suite compile --launch --timeout 180 --output <新しい証拠directory>
```

既に起動しているEditorへ`--launch`を送っても、既存processの環境は補完できない。環境不足で起動したことを確認できた場合に限り、保存済み状態と旧PID終了を確認してから起動する。一般の検証では`--launch`を付けず、同じ生存PIDを追跡する。

新Editor PID 139424はlaunch / compile成功、compile Error 0 / Warning 0、Editor内の`ALLUSERSPROFILE=C:\ProgramData`を確認した。起動環境の差を補った後の6回のPlay / StopでPackage Manager Errorは再現しなかった。既存package cacheを利用した同じconsumerであり、新規依存解決や全Unity契約の再実行とは扱わない。

## 空sceneと未準備sample UI

Unity 6000.3.16f1 / Sentis 2.6.1、active build target Androidだが、実行対象はWindows Editor / Direct3D12。Android Playerの結果ではない。sampleはGit `01f3d86`由来で、今回の文書ブランチは配布コードを変えていない。

NativeLeakDetectionをEnabledWithStackTraceにし、Game Viewを表示して各条件を3回Play / Stopした。各回の前にConsoleを保存・clearし、終了後はstack付き全ログを回収した。モデルの準備・推論は行っていない。

| 条件 | 実状態 | 終了後Console |
| --- | --- | --- |
| 空scene 1〜3回 | root 0 / sample 0 / fontなし | 各0件 |
| sample UI 1回目 | sample 1 / ready false / preparing false / font生成済み | 0件 |
| sample UI 2・3回目 | 同上、モデル未準備 | 各`Deleting invalid font reference.` Warning 1件 |

font Warningのstackは`UnityEditor.ScriptReloadProperties:Load`。モデル推論なしの繰り返しUIで再現したため、推論を実行しなければ発生しない問題ではない。空sceneでは再現しなかったが、この比較だけでsample / Unity / uloopの責任箇所は確定できない。

Persistent allocation Logは今回の6回では再現しなかった。[以前の空scene記録](results/m2-stability-baseline-windows-20261010.json)は実モデルを実行済みの別Editor processで1件を観測しており、この履歴を維持する。native leak解消やallocation原因の特定とは扱わない。

追加の時点別probeでは、元のleak mode Enabledへ戻した後の次のPlay開始直後にfont Warning 1件と`Found N leak(s) from callstack` Log 15件が既に存在した。font生成状態の確認後とStop後も全16件のままで、Stop後に初めて出たログではない。15件中13件は非ゼロ、2件は0と報告する診断であり、entry数を固有allocation数へ換算しない。stackは大部分がunknown native / JIT addressで、責任箇所は未特定。モデル準備・推論はこの追加probeでも未実行。検出設定の復元と同じprocessでの過去の状態が影響する可能性は未検証で、6回の0件からnative leak解消を推論しない。完全な最初の診断ログを結果JSONへ保存した。

PlayのCLI観測時間は空scene約4.71〜4.98秒、sample UI約4.74〜5.12秒。Stopは約0.57〜0.68秒。dispatcherや観測待ちを含むため、domain reload単体の時間や高速化達成とは扱わない。最終状態は元TextSearch scene / dirty false / Play停止 / compile停止 / leak mode復元。Editor本体は1つ。

## 次の作業

1. モデル未準備UIの2回目を使い、Play開始のreload前後をさらに分離して最小再現を固定する。fontのinstanceとGUIStyleの参照・Dispose前後・Editor reloadのタイミングを観測する。今回の時点別probeではWarningがStop前に存在することまで確認済み。
2. 失敗する最小テストを作り、font所有権や参照解放の必要な変更だけを実装する。警告を隠したりfontを解放しない変更で合格にしない。同じ実Editorの繰り返しUIとglyph / 検索 / 解放を確認する。
3. allocationはCPU / GPUのモデル実行と解放を別条件で観測し、font修正や今回の0件と混同しない。domain reloadの重さもimport / compile / Play切り替えごとに測る。
4. 監査済みAPKは接続ARM64端末で2起動・jar cold / warm・固定全CPU / GPU条件を実測する。Mac / iOS / sample Player / tag導入 / 正式Release gateも維持する。
