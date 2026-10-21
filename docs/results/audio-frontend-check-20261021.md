# 音声sampleの入力検査

公開tag `v0.1.0` のSDKを導入したconsumerで、追加の入力検査を実Editorで実行しました。
実行SDK sourceは `91e3419571432dfa54e7bbab045ef63a7cef4a81`、照合先は `643274033e561063439a3e41333ea8cfc5c13707` です。SDKへのsource overlayはありません。

元の合成fixture `padded_noise` の有効134400 sampleを `.f32` としてsampleの実readerへ渡し、有効839 frameと、元の `PlayerMediaInput.DecodePcm` が返す波形・マスクの全bytes一致を確認しました。
元のPCMとmaskは保持しています。元PCMのpaddingには正負のzeroがあり、既存デコーダーは有効prefixから正のzeroでpaddingを再構成します。この既存の入力契約との一致であり、元の数値参照・許容差や任意入力の切り詰めを変更した結果ではありません。

NUnitは1件合格・失敗0・skip0、processは正常終了しました。モデルのloadやGPU推論は実行していません。音声GPU反復の合格には数えません。

## 元bytesと実行記録のSHA256

| 対象 | SHA256 |
| --- | --- |
| 元PCM | `ca189983f2a78b942e20d70ddaf77b71aade9eb75ccc7c6b645f764427cc0e91` |
| 元sample mask | `61bd8a396bb9df446e6c4c8e8684428147150c79f159a2194469a94b7a3971ce` |
| `.f32` の有効prefix | `a78defcfa0d000eecf8f5eca429135170bb9568f8f4b68d8e14e2e36fae62949` |
| 既存decoderのcanonical波形 | `8d848b67e63d49e9a8a060dea446b925bb7a40080087bbbad17248df73afd38d` |
| 実行report | `361562513b5687f04769676098e1228afc002f589d85e0c9472a9d7bb7f2d651` |
| NUnit XML | `437a9400544f4bf86e7ec9772cbcac1779fb4b7b4bf3d6b0fb8b881e7ffccf18` |

原本のinput、report、XMLと追加検証harnessは非公開の検証領域に保持しています。この記録はモデル実行・検索品質・GPU互換性を追加で資格付けしません。
