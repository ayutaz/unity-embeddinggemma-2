# Windows GPUメモリの継続観測

Windows先行リリースの性能確認では、短い前処理・解放区間にも観測を増やすため、
Python toolsの `native_gpu_memory.PdhGpuMemoryQuery` で同じPDH queryを再利用します。
SDK RuntimeやSentisの推論backendは変更しません。重い実モデル試験の前に、
[小規模TDDと実Windows入口](results/native-gpu-memory-collector-tdd-20261020.json)を確認しました。
後続の[実モデル10条件の再照合](windows-media-memory-results.md)では、40出力・正常終了と
全180区間のsampleを確認しました。最低要件・正式公開の合格は別に残っています。

対象は明示したPIDのprocess lifetimeだけです。読み取り専用handleを保持し、
作成時刻と収集前後の生存を確認するため、PIDの再利用を別processの結果として
取り込みません。呼出側も、対象PIDが自分の起動した検証processであることを確認します。

`GPU Process Memory` のDedicated / Shared Usageを単一queryで収集します。
両counterの型、完全なPID / adapter名、status、整数byte数、重複、adapterの対応を
検査します。不在・不一致・失敗を0 byteとして扱いません。元のraw値と時計を保持し、
宣言したlocal UTC offsetで変換した時計が実際のUTC収集区間へ収まることを検査します。
この照合の1ミリ秒marginは時計の精度保証ではありません。

Windowsの[raw counter構造体](https://learn.microsoft.com/en-us/windows/win32/api/pdh/ns-pdh-pdh_raw_counter)は
provider値・status・local FILETIMEを返し、
[raw counter配列API](https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetrawcounterarrayw)は
必要なbufferサイズを取得してから配列を読む方式です。英語counter名を登録し、
収集のたびにPowerShellやqueryを起動しません。観測間隔の実績、収集overheadと
欠測区間も保存し、observerの追加を推論の高速化として数えません。

返す値はprocess全体の**観測したallocation**です。SDK専有allocation、容量、
保証されたpeak、GPU演算時間、Sentisでの実モデル成功を示しません。
load / first / warm / 各stage / releaseの実区間との対応と欠測を、実実行後に
`windows_gpu_memory.summarize_gpu_memory_intervals` で確認します。短い区間を
補間して埋めたり、採取できなかった値を最低要件の根拠に使ったりしません。

ローカルの原本・数値capture・生ログは非公開で保持し、公開する要約は
履歴・実payloadのprivacy確認を通します。新しい画像や録音はこの観測には不要です。
