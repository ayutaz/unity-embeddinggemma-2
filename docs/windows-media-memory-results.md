# Windows画像・音声のGPU性能とメモリ観測

2026-10-20の[元データ再照合](results/native-GPU-media-memory-readback-20261020.json)は、
既存の10条件・各4回の実推論、最終768次元vectorの独立照合、正常終了と解放を対象にします。
前処理・encoder・backboneの実値をそのまま接続し、元の参照と許容差を維持しています。
この記録の作成時にモデル変換・ダウンロード・推論は繰り返していません。

SDK実行sourceは `7754ce4`、検証sourceは `2cc7d94` です。
照合先 `8ef84a5` とのSDK差分はMarkdownのみで、Runtime treeは同一です。
各processの元report / source / XML / exit receipt / log、40出力、メモリ原本を再照合し、
各条件の原本保管2コピー、計20コピーも再hashしました。
現在のSDK sourceで再buildした実行や、新しいtag consumerの証拠には数えません。

実行構成はWindows、Unity 6000.3.21f1、Sentis 2.6.1、
NVIDIA GeForce RTX 4070 Ti SUPER、Direct3D12、driver 32.0.16.1088です。
CPU時間はUnity process全体、GPU timestampはqueueやcopyを含むsubmission範囲です。
全workerのfallbackを新たに検査した試験ではないため、GPU接続の機能受け入れには
[source別の既存検証](results/windows-first-stages-1-to-3-20261020.json)も必要です。

## 固定入力での観測結果

同じPNGまたはPCM / maskを各条件で `first`、`warm`、`repeat-1`、`repeat-2` の順に実行しました。
warm欄は後半3回の範囲で、入力の異なる仕事量や別processとの平均ではありません。
FP32の最終cosineは0.999以上、Float16重みは0.99以上、全vectorは有限・単位長条件を通過しました。
各stageとroundの元値、input / descriptor / referenceのhashはJSONに保持しています。

| module / weights | load 秒 | first 秒 | warm 3回 秒 | process host peak GiB | sampled dedicated GiB | sampled shared GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| audio-1119 / fp32 | 54.164 | 8.407 | 5.458–6.789 | 8.838 | 9.569 | 0.897 |
| audio-1119 / float16-weights | 51.049 | 7.165 | 5.237–5.497 | 7.789 | 10.038 | 0.569 |
| audio-839 / fp32 | 57.238 | 8.298 | 5.232–5.835 | 8.824 | 9.538 | 0.896 |
| audio-839 / float16-weights | 49.959 | 6.658 | 5.198–5.447 | 7.864 | 10.038 | 0.572 |
| image-square / fp32 | 38.602 | 6.239 | 4.623–4.949 | 8.844 | 4.913 | 0.905 |
| image-square / float16-weights | 46.798 | 7.639 | 4.676–5.150 | 7.614 | 5.413 | 0.578 |
| image-landscape / fp32 | 39.066 | 8.935 | 5.253–5.375 | 8.869 | 4.975 | 0.906 |
| image-landscape / float16-weights | 42.350 | 6.441 | 4.718–4.909 | 7.675 | 5.413 | 0.580 |
| image-portrait / fp32 | 45.359 | 6.967 | 5.098–5.305 | 8.830 | 4.975 | 0.906 |
| image-portrait / float16-weights | 39.639 | 6.919 | 4.752–5.031 | 7.738 | 5.350 | 0.581 |

loadにはmodel準備、firstには初回の準備・dispatch等を含みます。
各processのhost peakと、各区間のdedicated / shared最大値は別の観測です。
dedicated最大とshared最大が同時に発生したとは限らず、加算してpeakとは扱いません。
使用した16 GiB host監視上限、起動前のGPU空き容量基準もSDKの最低要件ではありません。

## 収集範囲と欠測

同じ[継続PDH query](windows-gpu-memory-observation.md)で対象processの作成時刻・生存・時計を検査しました。
10条件それぞれでload、release、12 stage、4 stage-spanの18区間、計180区間に有効なsampleがあります。
stage-spanはstageと重なるので、180の独立した仕事として合算しません。
最も短いreleaseでも有効sampleが2件あり、観測された最大sample間隔は17.618ミリ秒でした。

原本には182,077回のquery試行があり、182,065回のquery成功のうち168,903回でallocationが取得できました。
13,162回はqueryが成功しても対象GPU counterがまだ利用できない等の欠測です。
残る12回はquery errorです。両方を欠測として保持し、0 byteや補間した値には置換していません。
processが終了するまでobserverが動作し、繰り返す収集失敗の上限には達しませんでした。
全区間にsampleがあることは、区間の全瞬間を観測した証明ではありません。

観測はEditor process全体のallocationです。SDK専有量、物理VRAM容量、保証された最大値、
解放直後に0になる保証を示しません。closed workerのfield検査とprocess正常終了は別の証拠です。
CPU backendとの同条件比較、最低hardware要件、正式Release、新tag導入の合格は残っています。
過去の短区間の欠測・失敗原本は保持し、この新しい観測で過去の結果を書き換えません。

公開JSONは選択した数値・相対名・source / hashのみです。元log・path・capture・query原本は非公開で保持します。
