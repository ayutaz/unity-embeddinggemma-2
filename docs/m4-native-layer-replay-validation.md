# 固定入力による音声layer 0内部参照

2026-10-14 JST。現在の保存Sentis encoderではsilenceのlayer 0から固定許容差を超えるため、同じPython producerのsubsample出力を入力にして公式layer 0だけを再実行する診断producerを追加した。Runtimeや元参照・許容差は変更しない。正式Release / tagは作成しない。

## TDDと監査

selected loader / hook観測の初期5 assertion redはcommit `c666663`に保持。今回、既存5 greenとproducer不足3 assertion redを確認し、入力・最終値のsize / SHA、source・revision・許容差、snapshotのconfig / model hashを監査するproducerを実装して8 green。source / 許容差 / weight変更と、監査済みでも再現しないfinalを拒否する4条件を追加して合格した。CLI不足2 assertion red → 14 green、診断workflow不足1 assertion red → 1 green。

Pythonはtools/からuvで既存環境を再利用した。初回はuv cache、次はpytestの既存ユーザーtempへの権限不足で実行が失敗した。これらはTDDのredに数えず、workspace内のcacheと新しい専用basetempで上記のassertion失敗・合格を確認した。ローカルで実weightのdownloadや実モデル推論は行っていない。

loaderは公式`Gemma4AudioLayer`1層だけを構築し、safetensorsの対応するkeyだけを読みstrict loadする。全audio towerを構築しない。公式のposition encoderとbidirectional / blocked maskを利用する。この診断はfully valid silence profileのlayer 0専用で、partial mask・他信号の受け入れを証明しない。

hookは自身が登録したものだけをfinallyで削除し、外部hookと固定入力を保持する。再実行した最終値が保存済みnative finalとbit単位で一致した場合だけ7つの内部値を書き出す。一致しない場合はCLIが別reportへerrorを保存し非zeroで終了する。成功reportでも`full_encoder_validated=false` / `unity_validated=false`を保持する。

## Actionsで実参照を取得

専用label `audio-layer-replay-diagnostics`を付けた同一repositoryのPRだけで実行する。contents / actions read権限、forkの実行を除外し、secretは使わない。既存の全audio export workflowを同時に起動するlabelは付けない。

固定artifact `11814567883` / producer `4edbe17a8105233c5aaa1514faa2862b08a4f7e0`のreference / validation JSONとnative stage 00 / 01だけをrange取得する。JSONの既知SHAを確認してから、同revisionのconfig / modelだけをActionsで取得する。新しいproducer sourceと元参照sourceを分ける。失敗時もreport / range receiptをuploadし、failureをcontinue-on-errorで成功にしない。weightsはuploadしない。

## 実モデルActions結果

[Draft PR #75](https://github.com/ayutaz/unity-embeddinggemma-2/pull/75)、head `0908bc04b623d0facbc724c90c9861f178c59274`の[run 38459688048](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38459688048)は成功し、2026-10-13 20:04:16 UTCに完了。実行sourceはGitHubのmerge commit `9800c52957478fafa1377e2caa215f9f2e1477b0`で、元producer `4edbe17…`と区別する。[小さいreceipt](results/m4-native-layer-replay-actions-20261014.json)へ各source・各内部値のhashを固定した。

7,486,823,730 bytesの元archiveから4ファイルだけを2,383,011 bytes / 16 range requestsで取得。既知JSON SHAと各binaryを監査した。実モデルのconfig / safetensorsはActionsで取得して固定SHAを照合し、公式layer 0だけを構築した。実際のTorchは2.14.1+cu130 / threads2であり、envの4を実thread数として記録しない。

fully valid silence入力`[1,280,1024]`から再現したnative layer 0 finalは、保存済みfinalの全要素とbit単位で一致した。7つの内部値を各1,146,880 bytesで取得し、ダウンロード後も全7ファイルのsize / SHAを再照合した。norm_outのSHAは元finalの`3cc35396…`と一致。Runtimeの修正、元goldenの置き換え、許容差の緩和はない。以前の全audio export用workflowは条件によりskipで、不要なfull exportを実行していない。ここでのskipは数値合格には含めない。

これは公式native内部参照の再現成功であり、Sentisの内部値との照合はまだ未実行。encoder数値gateの不合格は保持する。source一致と正常完了が確認できても、元6信号・CPU / 実GPUの修正合格、画像数値差、各role・境界・sample・Player / 必須実機・最終配布準備は未完了。

同じheadの[通常CI 38459687450](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38459687450)はPython4環境 / package / actionlintが合格した一方、model-referenceは`audio ONNX export mismatch: silence`で失敗し、Required CIも不合格。native内部参照producerの成功で、元ONNXの失敗を置き換えない。診断完了後は専用labelを外し、文書更新で同じ参照取得を重複実行しない。後続headのCIとこの固定sourceの実測は別に追跡する。
