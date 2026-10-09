# M1 保存・量子化・測定と完了監査

実行完了した計画の履歴（2026-10-09更新）。作業ブランチは `feat/m1-persistence-benchmark`。
Unity 6000.3.16f1 / Sentis 2.6.1、固定モデルrevision・batch 1 / length 128 / 768次元で実行した。
実測コード `329da92` で全段階の受け入れ条件が合格し、PR #4をmain `8146107`へ統合済み。[検証記録](m1-completion-validation.md)を参照。
以下は実施した順序と合格条件。次の残作業は [M2計画](m2-plan.md)に整理する。

| 順序 | 実装・実行 | 合格条件と証拠 |
| --- | --- | --- |
| 1 | 保存APIの契約をテストし未実装redを確認。保存先検証・一時ファイル・再読み込み・置換を実装 | 小さいモデルで保存前後一致、入力モデル非破壊、fp16 round trip、不正引数・欠落・破損拒否 |
| 2 | 実モデルのfp32を保存・ロードし、CPU / GPUCompute各15件で元モデルおよびPython参照と比較 | 全件cosine >= 0.999、保存前後の最大絶対差 <= 1e-5、tokenizer完全一致、finite・shape・norm |
| 3 | コピーしたモデルにSentis Float16重み量子化を適用し、別の`.sentis`を保存・ロード | 全件cosine >= 0.99、CPU / GPUCompute実行、skipなし。サイズとSHA-256記録 |
| 4 | 同じ固定入力順で初回・warmup 2周・測定3周を各backend / precisionで実行 | 最終ベクトルのreadback完了を含むwall time。各case・中央値・p95、ロード・tokenizer準備・Worker作成・保存・量子化時間を区別 |
| 5 | CPU / GPUメモリと環境を記録し、全証拠を監査してPRを統合 | Unity全体のallocated / reserved / graphics-driver bytes、process working set / private bytesを段階別に取得。モデル専有やピークと誤記しない。未取得は未測定。小さい結果JSONと手順をcommit |

モデル生成は既存CI成果物を利用し、ローカルで重み取得・Python変換を再実行しない。実GPU検証と測定には既存Editor / uloopを使う。
保存・量子化は[Sentis標準API](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/quantize-a-model.html)を使用。
Float16は重みの保存形式であり、全演算がfp16になるとは扱わない。メモリは観測範囲を明記する。
Pythonの監査・ハーネス変更はtools/のuvとTDD、CI matrixで検証する。GPU skipや未実行を合格にしない。
M1完了はWindows Editor限定。他OS・モバイル・UPMリリースはM2、画像はM3、音声はM4に残す。
