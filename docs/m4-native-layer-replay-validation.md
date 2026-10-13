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

実モデルのActions結果はまだ未取得。内部参照の採用には固定finalの完全一致が必要で、未実行を成功にしない。参照取得後もSentis内部値との対応・原因演算修正、元6信号のCPU / 実GPU、画像数値差、各role・境界・sample・Player / 必須実機・最終配布準備が残る。
