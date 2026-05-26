可以。先講結論：**public 分數高的作法，其實都不是單靠更強 reranker，而是更好地做 top-5 coverage**。因為這題是 `Recall@5`，只要 gold evidence 有進前五就算，所以真正有效的是「把不同訊號互補起來」，不是把第 1 名排得更準。

## Public score 較高的方法總表

| Submission | Public score | 核心想法 |
|---|---:|---|
| `submission_memory_hybrid_rerank.csv` | `0.76194` | BM25+dense+gold memory+rereank |
| `submission_memory_hybrid_mxbai.csv` | `0.76887` | 上面再換成 `mxbai` reranker |
| `submission_dual_reranker.csv` | `0.77041` | 兩個 reranker + base rank 做 RRF |
| `submission_docaware_tune1.csv` | `0.79969` | supervised doc-aware ranker |
| `submission_rrf_docheavy3.csv` | `0.80200` | submission-level RRF ensemble |
| `submission_gate_seen_docheavy3.csv` | `0.80277` | 只在 seen-doc 啟用 doc-heavy ensemble |
| `submission_gate_nondescriptive_sim15.csv` | `0.80277` | 在 best public 基礎上，只對 non-descriptive 換成 `sim15` |

下面我按重要性說明。

---

## 1. `memory_hybrid_rerank` -> `0.76194`

實作主幹在 [HW3_112550043.py](/home/mace6728/INLP_HW3_112550043/HW3_112550043.py:1385)。

### 做法
每個 candidate 同時吃幾種分數：

- `BM25`：抓字面 match
- `dense retrieval`：抓語意相似
- `memory score`：看這個 evidence 在 train 的 gold evidence memory 裡，有沒有類似的 doc/content/page/type pattern
- `similar-question memory`：同 doc 的相似問題會不會投票給這個 candidate
- `page/layout/type` 額外 boost
- 最後再用 reranker 微調 top candidates

### 為什麼有效
這是第一個明確利用你之前發現的事實的版本：

- test 和 train 有很多 `doc_name` 重疊
- image evidence 其實能靠 `img_description` 當文字一起抓
- 單靠 lexical 或 dense 都不夠，memory 可以補一塊

### 為什麼只到 0.76
因為它還是**固定權重融合**。
也就是說，模型不知道什麼時候該信 memory、什麼時候該信 semantic。對 seen-doc 會有幫助，但也很容易在 OOD 題目上過度相信 overlap 訊號。

---

## 2. `memory_hybrid_mxbai` -> `0.76887`

### 做法
架構和上一支差不多，但 reranker 換成 `mixedbread-ai/mxbai-rerank-large-v1`。

### 為什麼比前一版高
`mxbai` 在這題有兩個優勢：

- 對 question 和 evidence 之間的細粒度語意對齊比較好
- 對表格/圖表的 `img_description` 文字也比較能抓到重點

所以它能把一些本來排在第 6 到第 10 的真 evidence 推進 top-5。
對 `Recall@5` 來說，這種改善是有效的。

### 為什麼提升幅度有限
因為 reranker 只能在**候選已經差不多對**的前提下幫忙。
如果 base score 本身因為 memory 偏掉，reranker 只是把錯的集合排序得更漂亮。

---

## 3. `dual_reranker` -> `0.77041`

實作主幹在 [HW3_112550043.py](/home/mace6728/INLP_HW3_112550043/HW3_112550043.py:1513)。

### 做法
先做一個 base ranking：

- BM25
- dense
- memory
- similar-question memory

然後用兩個不同 reranker 各自產生一份排序，再跟 base ranking 一起做 `RRF`（reciprocal rank fusion）。

### 為什麼會比單一 reranker 高
這題的 metric 是 top-5 coverage，所以：

- reranker A 抓到一部分 gold
- reranker B 抓到另一部分 gold
- base rank 還保留一些穩定 lexical/memory 命中

RRF 的好處不是讓第一名更準，而是讓**不同模型命中的 evidence 聯集更完整**。

### 為什麼沒有大幅超過 0.77
因為這一層還是在做「排序融合」，不是在解決真正的 distribution shift。
也就是說，它改善的是 candidate ordering，但 public 掉分更大的地方其實是 OOD 和 overlap 訊號誤用。

---

## 4. `docaware_tune1` -> `0.79969`

這是目前最重要的一次架構升級。
主體在 [HW3_112550043.py](/home/mace6728/INLP_HW3_112550043/HW3_112550043.py:1853) 和 [HW3_112550043.py](/home/mace6728/INLP_HW3_112550043/HW3_112550043.py:2092)。

### 做法
不再手動固定加權，而是把每個 candidate 轉成 feature，交給 `HistGradientBoostingClassifier` 做 supervised ranking。

feature 包含：

- bm25 / dense 原始與 normalize 分數
- memory score
- precise memory score
- similar-memory score
- page / layout context score
- neighbor vote：
  - exact
  - page
  - layout
  - page+type
- 各種 memory precision feature
- modality / raw_type / token length
- rank percentile
- doc overlap confidence

最後用：

- `model score * 0.8`
- `base score * 0.2`

再加上 `mxbai` reranker 做最後 refine。

### 為什麼 public 大跳
這一版的本質差異是：

> 它不是問「memory 要不要加 0.5」
> 而是在學「什麼情況下 memory 是可信的」

所以它能比 fixed-weight hybrid 更好地處理：

- seen-doc 但 overlap 很弱
- same-doc similar question 有票，但 semantic 本身不夠強
- page/layout 看起來像，但內容其實不對

### 為什麼還是卡在 0.79969
雖然它是最強單模型，但本地 `mixed` split 還是太樂觀。
後來你也看到：`mixed` 可到 `0.867+`，public 卻只在 `0.80` 左右。主因是 **domain OOD**。

---

## 5. `rrf_docheavy3` -> `0.80200`

### 做法
這不是新模型，而是 **submission-level ensemble**。
把幾支強模型的 top-5 做 RRF，但更偏向 doc-overlap / memory-heavy 的分支，所以我叫它 `docheavy3`。

大致上融合的是：

- `docaware`
- 舊的 dual reranker 類
- 舊的 mxbai / memory-heavy 類

### 為什麼比 `docaware_tune1` 高
這題最有效的 ensemble 方式不是平均所有東西，而是讓**不同模型各自補到別人漏掉的 evidence**。

`docaware` 很穩，但有些 seen-doc 題目，舊 memory-heavy 模型反而會把某些 gold evidence 放進 top-5。
RRF 正好能把這種互補性吃進來。

### 為什麼只小升
因為全域 ensemble 很容易把 OOD 題目也一起污染。
所以它有幫助，但還不夠精準。

---

## 6. `gate_seen_docheavy3` -> `0.80277`（目前 best public 之一）

### 做法
這是目前 best public 的關鍵想法：

- 如果 test row 的 `doc_name` **出現在 train**
  - 用 `docheavy3` 那種 overlap/memory 較強的融合結果
- 如果 `doc_name` **沒出現在 train**
  - 保留比較穩的 base submission，不硬套 overlap 策略

### 為什麼這一版最好
它抓到一個很重要的事實：

- overlap 訊號 **不是全域都好**
- 只在 **seen-doc** 上才比較值得信

所以這一版的進步不是「分數算得更精細」，而是**把 overlap 訊號用在對的地方**。
這也是為什麼它能從 `0.80200` 再升到 `0.80277`。

### 為什麼這個方向對
因為它和 public 的真實分布比較一致。
你之前已經分析出來：test 裡面既有 seen-doc，也有一大塊 OOD domain。這種 gating 比全域融合合理得多。

---

## 7. `gate_nondescriptive_sim15` -> `0.80277`（和 best tie）

### 做法
這支是後來的新嘗試：

- 先把單模型 `docaware` 改成 `sim15`
  - 也就是把 `similar_memory_weight` 從 `0.35` 降到 `0.15`
- 然後不是全域替換，而是只對 **non-descriptive** 題型，把 prediction 換成 `sim15` 版本
- 其他題目保留 `submission_gate_seen_docheavy3.csv`

### 為什麼本地 val 很強
在本地 mixed split，它是我目前最強的 routing：

- `docaware_tune1`: `0.867378`
- `docaware_sim15`: `0.875634`
- `gate_nondescriptive_sim15`: `0.879878`

代表 `sim15` 對：

- Comparative
- Interpretative
- Inferential
- 非 Descriptive 類

確實有幫助。

### 為什麼 public 只追平，不超過
因為它修掉的是**本地 split 上的題型偏差**，但 public 真正的主要痛點是：

- Academic paper OOD
- mixed-modality
- gold evidence 較多的題目
- seen-pattern 不足以泛化的 case

所以它能提高 local val，但不一定能打中 public 的主要失分區。

---

## 這幾個高分方法的共同特徵

真正有用的 pattern 其實很一致：

1. **把 image evidence 當文字處理**
   - 用 `img_description` 和 text evidence 一起排
2. **同時保留 lexical + semantic**
   - BM25 和 dense 缺一不可
3. **利用 train gold memory，但不能全域亂用**
   - 一定要看 seen-doc / overlap confidence
4. **用 ensemble 補 coverage**
   - 這題重點是 top-5 命中集合，不是 top-1 排序
5. **最好的提升都來自「何時該信哪個訊號」**
   - 不是單純把每個訊號都加大

## 我的判斷

如果只看目前 public，最成功的不是某個超強單模型，而是這個組合：

- `docaware` 當主幹
- memory-heavy 分支當補充
- 只在 seen-doc 或局部條件下啟用補充分支

也就是說，**高分方法的本質是 conditional fusion，不是更重的 reranking**。

如果你要，我下一則可以直接把這些方法整理成：
- 報告可直接貼的正式英文版
- 或者畫成一個「方法演進圖」讓你更容易講清楚。