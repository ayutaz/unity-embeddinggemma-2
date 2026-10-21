# Windows保守の中間確認

2026-10-21時点の工程1〜4の記録です。音声GPUの追加安定性検証は未実行で、全工程の完了記録ではありません。

## 1. 依存更新

元の[certifi #120](https://github.com/ayutaz/unity-embeddinggemma-2/pull/120)と[Pillow #121](https://github.com/ayutaz/unity-embeddinggemma-2/pull/121)は、それぞれlock内の対象1パッケージだけが変わることを独立照合しました。
registryはPyPI、配布先はPythonの公式package配布hostで、size / SHA256を持つ元のlock entryを維持しています。全配布物をこの確認で取得したという意味ではありません。

| 対象 | 元source | 元の完全なCI | 判定 |
| --- | --- | --- | --- |
| certifi 2026.7.22 → 2026.10.19 | `a787c56b80c123a7316f7a25db64bc35ae1c3441` | [39572778735](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39572778735) | Python4環境・model-referenceは成功。package privacyとRequired CIは失敗 |
| Pillow 12.3.0 → 13.0.0 | `c91335861e98d997ffed4997c9edfaeac232860a` | [39572791028](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39572791028) | Python4環境・model-referenceは成功。package privacyとRequired CIは失敗 |

共通原因は、Dependabotの生成commit-message trailerにある公式メール表記が、リポジトリの厳格な公開identity規則に適合しないことでした。私用メールと確認された事例ではありません。元PRの失敗を成功へ読み替えません。
privacyの例外を追加せず、元の依存差分だけを公開用identityとclean main祖先から[PR #123](https://github.com/ayutaz/unity-embeddinggemma-2/pull/123)へ再提案しました。元PRの履歴は書き換えていません。

新sourceは `3255645d9890ea92a73b6b68c0fee03f64153076`、変更ファイルは `tools/uv.lock`だけです。結合後の全package集合・対象外entry・lock metadataが元と同じことを照合しました。
lockのSHA256は `288e9983233a97b90ee77ab30c0a4ee051f447b0f1b321dc7a2058c878bb89d3`です。
このcheckpointでは[新sourceのCI 39602745551](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39602745551)は実行中で、まだmerge可能とは判定していません。

[Pillow 13の公式変更点](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html)と、このリポジトリのPython >=3.13・RGB画像操作を比較しました。削除されたI-mode PNG保存や型を変えるfromarrayのmode指定を、この生成経路は使っていません。
実Pillow 13.0.0で、変更していない既存の画像fixtureテスト関数2件を小さい隔離環境から呼び出し、元PNG bytesの再利用と異なる入力pixelsの保持を確認しました。
通常のfocused pytest起動は、その小さい環境にTorchがないためglobal conftestのimportで停止しました。full pytest合格とは数えず、完全な回帰はCIで確認します。
certifiは[公式リリース](https://github.com/certifi/python-certifi/releases/tag/2026.10.19)も確認しています。個別PRでの成功は、結合したsourceの最終受け入れを代替しません。

## 2. 音声GPUの追加反復

固定audio-839 / Float16の起動条件は、元の専用VRAM観測値と1,536 MiBの安全余裕に基づく**空き11,815 MiB以上**です。
この工程の最初の再確認では空き7,694 MiBで、条件未達でした。別のprocess観測ではWindows Terminalに7,553,593,344 bytesの専用allocationがありました。これらは別時刻の観測で、保証されたpeakではありません。
他の作業を含むTerminal終了の許可は未回答のため、processを終了せず、重い音声モデルも起動していません。CPU fallbackや余裕の引き下げはしていません。
再開時は容量・process寿命・所有関係を再確認し、実Playerと実sampleを順番に実行します。各実行30分以内、元数値gate・実Worker・順位とscore・キャンセル後の再試行・解放・正常終了を照合します。
画像・マイク・音声入力の過去の追加確認は、それぞれ元sourceの記録として保持します。今回の音声GPU反復の成功へ転用しません。

## 3. 結果と対応範囲

正式版の受け入れと公開後の追加反復を、[検証状況](../status.md)・[対応構成](../windows-supported-configuration.md)・[ロードマップ](../roadmap.md)で区別します。
Windowsを優先し、実機未実行の他OS、他GPU、最低メモリ、マイクの物理切断・復帰、実際のOS focus遷移を確認済みに追加しません。
音声の新しい実行結果が得られたら、この中間確認と区別した記録を追加します。元golden・許容差・生ログ・録音・モデルは変更・公開していません。

## 4. patch版の暫定判断

公開済みv0.1.0のsource `91e3419571432dfa54e7bbab045ef63a7cef4a81`と、保守確認source `f135d46cf6219599a739c44eb0961d9f2d041c79`で、SDK Runtime treeはともに `7d6e4314552b221d06db881405e7ccff3f449a77`、package版は `0.1.0`でした。
依存PR #123も変更はPython toolsのlockだけです。現時点ではSDKへの利用者向け修正がなく、これらだけを理由とする新しいSDK tag / Releaseは不要と判断します。
これは音声GPU工程の省略や全作業完了の判断ではありません。追加検証で修正が必要になった場合にpatch版を再判断し、影響範囲の回帰・実GPU・新規consumer・版 / CHANGELOG・固定tag・公開後の読み戻しを揃えます。
この確認でmerge、新しいtag、Release公開は行っていません。
