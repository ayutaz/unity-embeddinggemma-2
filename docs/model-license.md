# 配布コードと固定モデルのライセンス確認

確認日: 2026-10-10。

- このリポジトリとUPMコードのライセンスは [Apache-2.0](../LICENSE)。UPMには [LICENSE.md](../Packages/com.ayutaz.embeddinggemma/LICENSE.md) を含める。
- 対象モデルは `google/embeddinggemma-2`、固定revision `914f7f89142e33e77833254d9c9b90c3cef7303b`。
- [固定revisionのHugging Face metadata API](https://huggingface.co/api/models/google/embeddinggemma-2/revision/914f7f89142e33e77833254d9c9b90c3cef7303b) のshaが上記revisionと一致し、`cardData.license` は `apache-2.0` だった。[同revisionのモデルカード](https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/README.md) のlicense欄も一致した。
- [Googleの公式EmbeddingGemma説明](https://ai.google.dev/gemma/docs/embeddinggemma) でもEmbeddingGemma 2のApache-2.0を確認できる。リリース時には対象revisionと取得元を再確認する。

コードとモデルは、それぞれの一次資料で条件を確認する。モデル・生成した `.sentis` はGit / LFSやUPMに含めず、利用者が固定revisionから取得・変換する。
Player検証のbundleはモデルを含む内部の配置成果物であり、正式Releaseへの添付物ではない。
モデルを含むPlayer等を後で配布する場合は、Apache-2.0の条文、著作権・帰属表示、変更の表示を配布物に対応させて確認する。コードのLICENSEだけでモデルの帰属表示を済ませない。

この確認はM2のリリース準備の一部。Player / 実機精度、build、tag導入、正式Releaseは別の未完了gateである。
