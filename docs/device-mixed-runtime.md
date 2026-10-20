# Android / iOSの混在GPU検証入口

この入口は、端末用の固定fixture転送と、既存の全role・全module・混在検索の検証を接続します。専用build markerがあるPlayerだけで起動し、通常のtext・media・sample Playerの入口とは分けています。実機のGPU・操作・性能・権限・suspend / 復帰の合格を、この実装やEditorの契約テストから推定しません。

## 準備と起動

1. 公開の[モデル取得手順](media-model-preparation.md)で、レビュー済みFP32またはFloat16の6 module bundleを取得します。取得した`bundle.json`と`objects/`を、端末の`Application.persistentDataPath`を基準に`EmbeddingGemmaDeviceValidation/bundles/fp32/`または`bundles/float16/`へ配置します。これは既存のローカルprepared bundleです。このrunnerはモデルのダウンロード・変換を行いません。
2. 元参照・入力・画像の固定28ファイルとそのmanifestを用意します。fixture本体をbuild前に`Assets/StreamingAssets/EmbeddingGemmaDeviceValidation/fixtures/`へ配置します。この生成先はGit管理から除外します。元manifestの同一bytesを端末側の`EmbeddingGemmaDeviceValidation/fixture-manifest.json`へ配置し、その全SHA256を設定します。転送後も既存loaderが元参照・入力hashを独立に検査します。
3. 次の設定を端末側の`EmbeddingGemmaDeviceValidation/run.json`へ配置します。`fixture_manifest_sha256`は元manifestの実SHA256へ置き換えます。実パス・端末名・署名情報を設定や公開結果へコピーしません。

```json
{
  "schema_version": 1,
  "precision": "fp32",
  "phase": "cold",
  "run_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "cache_id": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "bundle_descriptor_sha256": "40aefa5de06f575d31b7582c24314c1ab035da374116b3ce3036de0e919cd647",
  "fixture_manifest_sha256": "<exact-original-fixture-manifest-sha256>"
}
```

4. 対象platformへ切り替え、固定sourceで`ValidationPlayerBuild.BuildAndroidMixed(output, codeCommit)`または`BuildIOSMixed(output, codeCommit)`を実行します。既存の可逆profileを使い、AndroidはARM64 / Vulkan、iOSはDevice SDK / Metal、両方ともRelease / IL2CPP / Highを設定します。iOSの署名は既存のXcode側で行います。署名情報は読み出し・変更・公開しません。
5. インストールして起動します。設定・bundleの端末への配置はADBまたはXcodeのアプリcontainer操作等、所有者が管理する方法を使います。この配置、APK / Xcode exportと実機起動はまだ未検証です。

Float16では`precision`を`float16`、descriptorを`8a6a90c5f656a00b024553a0412e5c4d32b4d2c9c7c5b3a0c2c146f557b4e27e`へ変更します。各起動の`run_id`は新しい32桁の小文字hexにします。warmは同じprecision・cache IDと、完了済みcoldのqualificationを使い、別processで新しいrun IDを指定します。既存reportやfixture転送先は上書きしません。

## 実行と証拠

### build前の固定入力配置

`embeddinggemma_tools.platform_ci` に、Android / iOS build用の明示的なfixture配置オプションを追加しました。認証設定済みの対応Macで、`tools/`から既存のnative build入口へ次を指定します。変数には実行者の私的な入力directory・元manifest・固定40桁checkout SHAを設定し、値を公開ログへ出しません。

```sh
uv run --no-project --python 3.13 python -m embeddinggemma_tools.platform_ci \
  --project .. --platform android --source "$SOURCE_COMMIT" \
  --output ../artifacts/device-build-with-fixtures --execute \
  --fixture-source "$FIXTURE_SOURCE" --fixture-manifest "$FIXTURE_MANIFEST" \
  --fixture-manifest-sha256 524ddbfceb05fc0d799e0b26a253f4755b82bfa304839e22bc6402d5724b295a
```

iOSでは `--platform ios` を指定します。3つのfixtureオプションは一緒に指定する必要があります。実checkoutの照合後、Unity導入・activation・buildの前に、manifest全bytesのpin、相対pathの重複・逸脱、全ファイルのサイズ / SHA256を検査します。symlink / Windows junctionを拒否し、既存のStreamingAssets内の利用者ファイルと専用directoryを上書きしません。入力の途中変更やコピー失敗では未完成fileを除き、検証済みの出力を保持してbuildを停止します。

配置先は `Assets/StreamingAssets/EmbeddingGemmaDeviceValidation/fixtures/`、同じ専用directoryの `fixture-manifest.json` は元manifestの同一bytesです。端末のpersistent dataへのmanifest・run設定・モデルbundle配置と、モデル取得は別に必要です。既定のhosted workflowはこの入力を自動取得せず、オプション未指定では `fixture_staged` はfalseのままです。

[TDDと元28ファイルの実配置](results/device-fixture-build-staging-tdd-20261020.json)では初回40件Red、workflow依存pathの1件Redを経て、関連128件Green・skip0です。元28ファイル・34,474,958 bytesを配置し、元入力・配置先・完全な私的2コピーのサイズ / hashを独立照合しました。配置の成功を元loaderの受け入れ、Android jar転送、実build、実機GPU・性能合格には数えません。SDK処理・元数値参照・許容差は変更していません。

### 端末での検証

専用runnerはEditor・Development build・computeなし・target不一致を入力読込前に拒否します。AndroidのVulkan、iOSのMetalを明示的に確認します。この確認は実行の入口条件であり、GPUの実行証拠ではありません。

固定manifestに従い、`file://`またはAndroidの`jar:file://`のStreamingAssetsを新しい`runs/<run_id>/fixtures/`へ転送します。全ファイルのsize / SHA256、途中ファイルのcleanupと元loaderの照合を維持します。通信先を指定する設定やCPU fallbackの設定はありません。

その後、既存の`PlayerMixedExecution`が画像3・音声2 moduleのquery / document / raw roleを順に処理し、textと18文書・4 queryの混在indexを検証します。36 base / 45 warm media vector、実GPU監査・数値gate・解放、別processのcold / warm cache条件を維持します。元のWindows入口も同じpolicyを使い、Editorや不一致targetの拒否を維持します。

端末内の結果は`reports/<run_id>-device.json`と`reports/<run_id>.json`です。前者は転送と呼出し、後者は既存の数値・GPU検証を記録します。公開前に実bytesをprivacy検査し、raw vector・ローカル診断・未レビューの画像は公開しません。入力・modelの実パスや端末のユーザー名をreportへ掲載しません。

`device_acceptance_completed`、`all_platforms_accepted`、`final_performance_accepted`は、この入口だけではtrueにしません。実機・OS・graphics・メモリ・permissions・suspend / 復帰・外部process証拠と独立監査が別途必要です。iOSは`Application.Quit`を無視する場合があるため、Quit呼出しを正常終了の証拠にはしません。macOS Editorの受け入れも別の未完了条件です。

[契約TDDと190件の関連回帰](results/device-mixed-runtime-ingress-tdd-20261019.json)は、最初の39件Red、入口・build検査を加えた48件Redから48件Green、初回の関連177件Green、StreamingAssets URIの追加13件Red→13件Green、最終190件Green・skip0・exit0を記録しています。すべてWindows Editor / no-graphicsの小規模確認であり、モデル推論や実機GPUの結果ではありません。
