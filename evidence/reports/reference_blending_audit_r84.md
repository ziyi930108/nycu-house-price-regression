# R84 參考混合與空間修正：有界來源盤點

本文件只解釋目前保存程式的機制，不選定下一個候選，也不改 R84。只讀來源及 development 已保存的小 JSON；未讀預測陣列／模型、未重算 MAE、未訓練。其他年度的結果與整體選擇不在此次盤點範圍。

## 現有觀察與真正入口

[development/results.json](../outputs/round84/development/results.json) 保存的 R84−v24 整體 MAE 差如下，負值較好：

| 比較層 | 差額（元） |
|---|---:|
| 原生 seed42，同 seed 4499 vs 2999 樹 | −31,739.0381 |
| 原生 seed137，同 seed 4499 vs 2999 樹 | −18,936.3534 |
| 原生兩 seed 等權平均／base | −24,475.4989 |
| reference | −865.9035 |
| spatial | −1,038.6477 |
| 最外 median | +930.0771 |

這些是保存的各層指標差，**不是可相加的因果貢獻**。原生平均的新案／舊案 MAE 差分別為 −9,830.0653／−50,881.3457；經 reference 後為 +7,181.0977／−15,374.7182，最終為 +2,708.7462／−2,276.8790。這提示影響並不均勻，但未證明原因。

[R84 driver](../src/round84_experiment.py#L330) 的 base 是 `B=.5*p42+.5*p137`，兩支都已是非負、未取整的 TWD；此處沒有舊 raw 或 LightGBM 子槽。R83 控制用相同後處理。[R48 的來源鏈](../src/round48_nn_experiment.py) 經 R39→R36→R35→[R12](../src/round12_experiment.py#L15)，實際呼叫的是 [round2_neighborhood.predict_reference](../src/round2_neighborhood.py#L321)，空間 helper 是 [round8_spatial_reference](../src/round8_spatial_reference.py#L177)。`round3_reference` 雖也匯入該函式，但其另外的城市 regional trend 並非這裡執行的方法。舊 reference docstring 的 `.75 raw+.25 unit` 是歷史說明，當前 R84 呼叫端明確傳上述純 native 平均。

## Reference：混合，而非全列換成參考價

令 `T=exp(beta_time*clip(query_day−fit_max_day,0,366)/365.25)`。未來日期來自 query X，beta、fit 最大日期及價格參考只來自保存的 fit reference；沒有 query y 參數。

[實作 236–303 行](../src/round2_neighborhood.py#L236) 先依保存群組順序取第一個可用層級；在特徵距離近鄰中，用 `1/(.05+d²)` 加權中位數估局部殘差，再加回 hedonic 項與建物面積得到局部 log 價格 `ell`。project 最多 16 鄰，其餘最多 40；reference/query 建物面積的固定適用範圍為 15–1500 m²。

```
c = w_level / (1+(nearest_distance/2)^2) * exp(−MAD/.18)
L = max(B,1) * exp(clip(ell−log(max(B,1)), log(.65), log(1.5)))
P = (1−c)*B*T + c*L                  # variant='trend'
```

[固定 w_level](../src/round2_neighborhood.py#L48)：project .75、address .65、street_section .40、locality .35、street .30、district_style .12。距離與離散程度再降低 c；它不是校準後的可信機率。局部價有相對 B 的 0.65–1.5 倍限制，所以 L 並非完全獨立於新 native 的固定常數。

無合適 reference、或不符合面積条件時 c=0，trend fallback 是 **B*T**，不一定等於 B。`anchor` 則用 `(1−c)*B+c*L`；兩者差在剩餘 base 的時間外推，不是把參考混合開／關。最終 reference 僅作非負保護、不取整。來源與 confidence diagnostics 由 X、保存 reference 決定，不依 B 或 query y。

## R39 保存的 spatial：只補較弱來源

[固定規則與公式](../src/round8_spatial_reference.py#L18) 只允許 `district_style`、`base_fallback`，並要求有效日期、15–1500 m²、有效地理與同城市／建物型態／主要用途、位置 extent≤2000 m。半徑 2000 m 內排除 query 同案，至少 5、最多 40 個 fit 建案，每案先取殘差中位數，再在建案間等權取 center 與 MAD。保存 state 使用既有 reference 殘差，不為 R84 重估。

```
Bfuture = B*T
G = clip(area*exp(center + hedonic@beta), .65*Bfuture, 1.5*Bfuture)
alpha = .30*exp(−project_MAD/.18)
U = P + (1−c)*alpha*(G−Bfuture)
```

G 的實作採 log 比值裁切，避免直接指數溢位；它也會隨 B 的裁切界移動。這是加在 P 上的殘差補充，**不是把 P 整體再乘一次 T**，也不是所有列固定 30% 混合。無足夠建案／不合格／Bfuture 無效時完整保留 P。development 小報告的控制與候選皆保存 mean reference weight 0.2428318、spatial applied 9,656、fallback 71,578／81,234 列；相同覆蓋不代表相同調整金額或 MAE。

## 最外 median：固定兩分支決定逐列邊界

[R84 第 350–354 行](../src/round84_experiment.py#L350) 固定 R71 NN 分支 N、R46 log 分支 H，輸出 `M=median(N,H,U)`。在實數運算上等價 `clip(U,min(N,H),max(N,H))`：U 在兩界內時原樣通過；同側越界的變動可能完全不傳到 M。保存 JSON 記錄單價 spatial 81,234 列改變，而完整輸出 28,242 列改變。

**限制預測差不代表限制 MAE 差。** 不同列的收益／損失可能被不同程度保留；即使 U 的整體 MAE 改善，M 仍可退步。沒有逐列誤差與邊界歸屬診斷，本次不能斷言「reference 不佳」、「median 過強」或把少掉的改善量當某層造成的損失。

## 未來可分離的介面，尚未立項

直接控制 reference 與 native 份額的既有設定是 `PARAMETERS['neighborhood_base_weights']`；不是 tree depth，也不是 `trend`／`anchor` 本身。若完整 R84 三期證據日後支持研究，只能先固定一個權重變更規則、保留 native/reference state/其他流程作配對比較，不在重用年度上搜索多組倍率。此處不選倍率或修改候選。

重要界線：c 同時進入 spatial 的 `(1−c)`，改 reference 權重也會改 spatial 剩餘份額；不能把結果宣稱為純粹的單層因果效果。舊 model/parameters 有完全一致檢查，未來需新版本化推論契約，不能就地修改已 pin 的 constants 或假稱保存 state 仍完全相同。空間 `maximum_weight=.30` 雖在參數表，predict 目前亦寫成 literal `.30`，僅改字典不會改實際算式。

任何後續比較須等 R84 三期完成再另立固定計畫；無新 blind holdout，反覆使用窗口與單次 GPU 重訓的限制仍在。更充分傳遞 native 變動可能放大高價尾端風險，不保證 Private 改善。本文件未改 source、CSV、selector 或任何推薦。
