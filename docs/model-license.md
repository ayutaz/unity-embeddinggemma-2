# 配布コードと固定モデルのライセンス確認

確認日: 2026-10-20。固定revisionのライセンス宣言と、公開モデルpreviewの法的表示ファイルを読み取り専用で再確認しました。[実応答の照合記録](results/model-preview-legal-readback-20261020.json)にURL・size・SHA256・producerと確認範囲を記録しています。Runtime・実機・性能の受け入れは[検証状況](status.md)を参照してください。

- このリポジトリとUPMコードのライセンスは [Apache-2.0](../LICENSE)。UPMには [LICENSE.md](../Packages/com.ayutaz.embeddinggemma/LICENSE.md) を含める。
- 対象モデルは `google/embeddinggemma-2`、固定revision `914f7f89142e33e77833254d9c9b90c3cef7303b`。
- [固定revisionのHugging Face metadata API](https://huggingface.co/api/models/google/embeddinggemma-2/revision/914f7f89142e33e77833254d9c9b90c3cef7303b) のshaが上記revisionと一致し、`cardData.license` は `apache-2.0` だった。[同revisionのモデルカード](https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/README.md) のlicense欄も一致した。
- [Googleの公式EmbeddingGemma説明](https://ai.google.dev/gemma/docs/embeddinggemma)も参考資料です。今回の再確認根拠は固定revisionのmetadata APIとモデルカードの実応答で、どちらもHTTP 200、metadataのrevisionと両方のApache-2.0宣言を確認しました。元応答の完全な私的2コピーを保持しています。

## 公開モデルpreviewの表示ファイル

FP32 / Float16重みの[実験用モデルpreview](media-model-preparation.md)では、モデル本体と別に`LICENSE.txt`・`NOTICE.txt`・`MODIFICATIONS.txt`を公開しています。両previewのproducer tagは`567befbd64287a706edaac59c978927e0277188a`に一致し、各825 assetのうち上記3ファイル、合計6つの実payloadを再取得しました。実size・SHA256は配信元metadataに一致し、全6ファイルの完全な私的2コピーとprivacy検査を確認しました。

利用者はbundleとともにこの3ファイルを取得・保持してください。`LICENSE.txt`のApache-2.0宣言を確認していますが、この限定照合はモデル全objectの再監査、あらゆる利用・再配布条件の判断、実GPU・必須実機・正式SDK公開の合格ではありません。モデル本体は今回再取得していません。

## コードとモデルの配布境界

コードとモデルは、それぞれの一次資料で条件を確認する。モデル・生成した `.sentis` はGit / LFSやUPMに含めず、利用者がpreviewを明示取得する経路と、固定revisionから取得・変換する経路を分ける。
Player検証のbundleはモデルを含む内部の配置成果物であり、正式Releaseへの添付物ではない。
モデルを含むPlayer等を後で配布する場合は、Apache-2.0の条文、著作権・帰属表示、変更の表示を配布物に対応させて確認する。コードのLICENSEだけでモデルの帰属表示を済ませない。

この確認は正式公開前の準備の一部です。Windows Playerの固定sourceでの機能検証と、未完了の必須他環境・性能・最終tag導入・正式Releaseを分け、[リリース計画](release-plan.md)の条件で判定します。
