# Full Guide：BrainPy Wendling Model（`wendling_sim`）

這份文件的目標是：**把你從「完全沒頭緒」帶到「能跑 single / network / whole-brain simulation，能寫自己的 loss（含多個 loss + 權重 + 很複雜的計算），也能客製化 optimization」**。

本專案刻意維持極簡 public API：你主要只會用兩個函式：

```python
from wendling_sim import simulate, optimize
```

如果你只記得一件事：**一切都是「config dict → simulate → features → loss → optimize」**。

---

## 0) 先把專案跑起來（最短路徑）

### 0.1 環境（建議用 repo 內已存在的 conda env）

本 repo 的 scripts 多數用：

```bash
conda run -n brainpy_model python ...
```

或你也可以：

```bash
conda activate brainpy_model
```

確認核心套件：

```bash
conda run -n brainpy_model python -c "import brainpy as bp; import nevergrad as ng; print('brainpy', bp.__version__, 'nevergrad', ng.__version__)"
```

把本 repo 安裝成可 import 的套件（建議；`requirements.txt` 末尾有 `-e .`）：

```bash
conda activate brainpy_model
pip install -r requirements.txt
```

如果你暫時不想 install，也可以跟 `scripts/*.py` 一樣把 `src/` 加到 `PYTHONPATH`/`sys.path`（見 FAQ）。

### 0.2 一行跑 single-node simulation（Python）

```python
from wendling_sim import simulate
from wendling_sim.model.params import get_type_params

res = simulate(
    sim_cfg={"dt_ms": 0.1, "duration_ms": 5000, "jit": True},
    model_cfg=get_type_params("Type4"),
    noise_cfg={"seed": 0},
)
print(res.lfp.shape)  # (T, 1)
```

### 0.3 用 CLI 跑（最適合新手）

- 跑單點 + 畫圖：

```bash
conda run -n brainpy_model python scripts/run_simulate.py --type Type4 --plot
```

- 跑 single-node 的 PSD fitting（nevergrad）：

```bash
conda run -n brainpy_model python scripts/run_optimize.py --target-type Type4 --budget 50
```

---

## 1) 你要先建立的「心智模型」：本專案到底在做什麼

你可以把 repo 當成 4 層管線：

1. **Simulation**：Wendling 動力系統（單點或 N-node 網路）
2. **Feature**：從模擬出來的 LFP 取特徵（目前主力：PSD）
3. **Loss**：把「模擬特徵」跟「target 特徵」算距離（可多個 loss + 權重）
4. **Optimization**：用 nevergrad 反覆提參數 → simulate → feature → loss，直到 loss 最小

對應的核心檔案：

- Public API：`src/wendling_sim/api.py`
- Simulation runner（回傳 `SimResult`）：`src/wendling_sim/sim/runner.py`
- Model（單點 / network）：`src/wendling_sim/model/wendling_single.py`、`src/wendling_sim/model/wendling_network.py`
- Connectivity（讀檔/產生/正規化）：`src/wendling_sim/connectivity/io.py`
- PSD：`src/wendling_sim/features/psd.py`
- Loss registry + 實作：`src/wendling_sim/loss/registry.py`、`src/wendling_sim/loss/psd_mse.py`
- Optimization loop：`src/wendling_sim/optimize/nevergrad_engine.py`
- Search space（參數 bounds）：`src/wendling_sim/optimize/search_space.py`

---

## 2) `simulate()` 全解：你要怎麼跑 single / network / whole-brain

`simulate()` 在 `src/wendling_sim/api.py`，實際上是把你給的 configs 交給 `run_simulation()`（`src/wendling_sim/sim/runner.py`）。

簡化後的介面：

```python
simulate(sim_cfg, model_cfg, network_cfg=None, stim_cfg=None, noise_cfg=None, monitor_cfg=None) -> SimResult
```

### 2.1 `SimResult` 會給你什麼

`SimResult` 定義在 `src/wendling_sim/sim/runner.py`：

- `t_s`: 秒（shape `(T,)`）
- `lfp`: **time-major** 的 LFP proxy（shape `(T, N)`）
- `states`: 你指定監控的 state（可選）
- `meta`: 各種 metadata（dt、duration、seed、n_nodes、G_net、connectivity_meta、params…）

你可以：

```python
res.save("runs/my_sim.npz")
res2 = SimResult.load("runs/my_sim.npz")
```

### 2.2 `sim_cfg`：時間、JIT、（目前 integrator 幾乎不影響）

`src/wendling_sim/sim/runner.py#_parse_time_cfg()` 會吃兩種單位：

- `dt_ms` / `duration_ms`（新手建議）
- `dt_s` / `duration_s`

常用鍵：

- `jit`: `True/False`（BrainPy DSRunner JIT；第一次會編譯，之後才快）
- `progress_bar`: `True/False`
- `debug`: `True/False`（沒有指定 monitors 時，會多記錄一些 states）
- `integrator`: `"rk4"` / `"euler"`（注意：**目前 model 的 `update()` 是手寫 Euler step，所以 integrator 選項實際上不會改變數值結果**）

> 你如果想要真的換 RK4/其它 integrator，通常需要把 model 改成 BrainPy 的 ODE system 寫法，而不是在 `update()` 裡自己做 `y = y + dt*dy`。

### 2.3 `model_cfg`：Wendling 的 local parameters（單點與網路都用）

所有參數的 default 在 `src/wendling_sim/model/params.py#STANDARD_PARAMS`。

你可以只傳你要覆蓋的部分，例如：

```python
model_cfg = {"A": 5.0, "B": 10.0, "G": 15.0, "p_mean": 90.0, "p_sigma": 2.0}
```

或直接拿六種 activity type 的 preset：

```python
from wendling_sim.model.params import get_type_params
model_cfg = get_type_params("Type4")
```

### 2.4 `noise_cfg`：seed、（network 才有 shared/mode）

在 `run_simulation()`：

- `seed` 會拿來設定 `brainpy.math.random.seed()` + `numpy.random.seed()` → 讓每次 run 可重現。
- `noise_cfg` 也可以覆蓋 `p_mean` / `p_sigma`（在 `src/wendling_sim/sim/runner.py#_prepare_params()`）。

注意兩個「目前的實作現況」：

1. **single-node**：噪聲是在 `src/wendling_sim/model/wendling_single.py` 裡固定用 `bm.random.randn(1)`（高斯），`noise_cfg["mode"]` 不會被用到。
2. **network**：`src/wendling_sim/model/wendling_network.py` 支援：
   - `noise_cfg["mode"]`: `"gaussian"`（預設）或 `"uniform"`
   - `noise_cfg["shared"]`: `True/False`（所有 node 共用同一個 noise sample 或各自獨立）

### 2.5 `monitor_cfg`：你想在 optimize 以外「抓更多內部狀態」就靠它

你可以要求 `SimResult.states` 存指定變數，例如：

```python
monitor_cfg = {"variables": ["y0", "y1", "y2", "y3", "y4"]}
```

注意：whole-brain（很多 node）時監控 states 會很吃 RAM。

### 2.6 `network_cfg`：你要怎麼「建立 network / whole-brain」

所有 connectivity 的 loading/generation 都集中在 `src/wendling_sim/connectivity/io.py#load_connectivity()`。

`simulate()` 會把 `network_cfg` 交給 `load_connectivity()`，然後：

- `n_nodes == 1` → `WendlingSingleNode`
- `n_nodes > 1` → `WendlingNetwork`

#### 2.6.1 最重要的語意：`W[i, j] = source j -> target i`

這是本 repo 的 locked decision（也寫在 `docs/WENDLING_Subnetwork_WholeBrain_SPEC_TODO_v0.3.md`）。

也就是：

```text
u_net = G_net * (W @ out)
out[j] 影響到 u_net[i] 的權重是 W[i, j]
```

#### 2.6.2 你可以用 4 種方式提供 connectivity

**A) 直接給矩陣**

```python
network_cfg = {"n_nodes": 20, "W": W, "G_net": 0.8, "normalize": "row_sum"}
```

**B) 從檔案載入（`.npy` / `.csv` / `.mat`）**

```python
network_cfg = {
    "n_nodes": 86,
    "W_path": "data/connectome/W.npy",
    "labels_path": "data/connectome/labels.txt",  # optional
    "normalize": "row_sum",
    "remove_self_loops": True,
    "G_net": 0.6,
}
```

`.mat` 的載入用簡單 heuristic：會優先找 key `W/sc/len/data`，否則抓第一個 ndim>=2 的矩陣（見 `src/wendling_sim/connectivity/io.py#_load_matrix_from_path()`）。

**C) 用 generator 自動產生（適合做 toy network）**

generator 實作在 `src/wendling_sim/connectivity/generators.py`，名稱包含：

- `erdos_renyi`
- `small_world`
- `ring_lattice`
- `stochastic_block_model`

使用方式：

```python
network_cfg = {
    "n_nodes": 20,
    "generator": {
        "name": "erdos_renyi",
        "options": {"p": 0.2, "weight_dist": "lognormal", "weight_scale": 0.8, "seed": 123},
    },
    "normalize": "row_sum",
    "remove_self_loops": True,
    "G_net": 0.8,
}
```

**D) 你自己寫 builder callable（最彈性）**

```python
def my_builder(n_nodes: int, seed: int = 0, **kwargs):
    # return W or (W, labels, meta)
    ...
    return W

network_cfg = {
    "n_nodes": 86,
    "builder": my_builder,
    "builder_kwargs": {"foo": "bar"},
    "normalize": "row_sum",
    "G_net": 0.6,
}
```

#### 2.6.3 `normalize` 與 `remove_self_loops`

在 `load_connectivity()`：

- `remove_self_loops` 預設 `True`（會把 diag(W)=0）
- `normalize` 預設 `row_sum`（把每個 target row 的總入流規一化，避免密集圖爆掉）

`normalize` 可用：`none` / `row_sum` / `max`。

---

## 3) PSD feature：你要怎麼把 simulation 變成「可被 fitting 的 target」

PSD 實作在 `src/wendling_sim/features/psd.py#compute_psd()`（Welch）。

### 3.1 最常用的 3 個模式（`roi`）

`compute_psd(lfp, ..., roi=...)` 支援：

- `roi="none"`：保留每個 node 的 PSD → `psd` shape `(F, N)`
- `roi="mean"`：先把 LFP across nodes 平均，再算 PSD → `psd` shape `(F, 1)`（或 `(F,)`）
- `roi="subset"` + `roi_nodes=[...]`：只算特定 node 子集

**新手建議先用 `roi="mean"`**，因為 loss 最簡單（1D）。

### 3.2 你自己的資料怎麼變成 target PSD

假設你有 `lfp`（shape `(T,)` 或 `(T, N)`）：

```python
import numpy as np
from wendling_sim.features.psd import compute_psd

lfp = np.load("my_lfp.npy")
dt_ms = 0.1
fs = 1000.0 / dt_ms

target = compute_psd(lfp, fs=fs, roi="mean", freq_range=(1, 50))
target_freqs, target_psd = target.freqs, target.psd
```

### 3.3 一個你一定會踩的坑：target 與 simulation 的頻率網格要對齊

Welch 的 `nperseg/fs` 不同時，`freqs` 網格會不同。

你有兩種作法：

1. **讓 target 跟 simulation 用同一套 PSD 設定**（最簡單）
2. 把 target PSD **插值**到 simulation 的 `freqs` 上（`scripts/demo_hcp_optimize.py` 就是這樣做）

---

## 4) Loss 全解：我到底要在哪裡新增？為什麼要 registry？怎麼 call？

### 4.1 為什麼要 registry？

你想要的使用體驗通常是：

- 在 YAML/JSON/命令列裡寫 `loss_name: psd_mse`
- 或寫一個 losses 清單：`[{name: A, weight:...}, {name: B, ...}]`

這時你不能在每個地方都 `import my_loss`，否則 config 根本沒意義。

所以用 `registry`（`src/wendling_sim/loss/registry.py`）的好處是：

- **loss 用字串選擇**（config-friendly）
- **可插拔**：你可以在不改 optimize pipeline 的情況下新增/替換 loss
- **runtime 擴充**：你可以在自己的 script 裡動態 `register_loss(...)`

### 4.2 內建 loss 有哪些（以及它們吃什麼）

內建 loss 在 `src/wendling_sim/loss/psd_mse.py`，並在 `src/wendling_sim/loss/registry.py` 註冊：

- `psd_mse`
- `weighted_psd_mse`
- `log_psd_mse`

它們的「共同介面」是：

```python
loss(psd, target_psd, freqs=..., **cfg) -> float
```

其中：

- `psd` / `target_psd` 可能是 `(F,)` 或 `(F, N)`（取決於 `roi`）
- `freqs` 是 `(F,)`

### 4.3 新增你自己的 loss：兩種方式（強烈建議先用 B）

#### A) 永久新增到 library（會進 repo）

1) 新增檔案：`src/wendling_sim/loss/my_loss.py`

```python
import numpy as np

def my_loss(psd: np.ndarray, target_psd: np.ndarray, freqs=None, **kwargs) -> float:
    return float(np.mean((psd - target_psd) ** 2))
```

2) 在 `src/wendling_sim/loss/registry.py` 註冊：

```python
from wendling_sim.loss.my_loss import my_loss
register_loss("my_loss", my_loss)
```

3) 用它：

```python
objective_cfg["loss_name"] = "my_loss"
objective_cfg["loss_cfg"] = {"any_kw": 123}
```

#### B) 在你自己的 script runtime 註冊（最適合快速迭代）

你可以像 `scripts/demo_hcp_optimize.py` 一樣：

```python
from wendling_sim.loss.registry import register_loss

def my_loss(psd, target_psd, freqs, band=(8, 13)):
    # 做任何你要的複雜計算…
    return ...

register_loss("my_loss", my_loss)
```

然後照樣在 config 用字串指定它。

### 4.4 多個 loss + 權重（你說的「好幾種 loss、有不同權重」）

`src/wendling_sim/api.py#optimize()` 支援：

```python
objective_cfg = {
  "sim_cfg": {...},
  "psd_cfg": {...},
  "losses": [
    {"name": "psd_mse", "weight": 1.0, "cfg": {}},
    {"name": "log_psd_mse", "weight": 0.2},
    {"name": "my_loss", "weight": 0.5, "cfg": {"foo": "bar"}},
  ],
}
```

內部會把它做成：

```text
total_loss = Σ weight_k * loss_k(...)
```

而且每次 evaluation 的 per-loss breakdown 會進 `OptResult.meta["loss_details"]`（見 `src/wendling_sim/optimize/nevergrad_engine.py`）。

---

## 5) Optimization 全解：我要從哪裡設定？怎麼客製化？

### 5.1 `optimize()` 在做什麼（極簡版）

`src/wendling_sim/api.py#optimize()` 做的事是：

1. 建 SearchSpace（bounds）
2. 建 objective_fn（candidate params → simulate → PSD → loss）
3. 交給 `NevergradOptimizer` 跑 ask/tell loop

### 5.2 你要調的兩大 config：`opt_cfg` 與 `objective_cfg`

#### `opt_cfg`（優化器層）

常用鍵：

- `budget`: 評估次數（越大越久）
- `optimizer`: nevergrad optimizer 名稱（例如 `NGOpt`, `CMA`, `DE`, `PSO`, `OnePlusOne`…）
- `num_workers`: 平行 worker（>1 才會平行 ask/tell；是否真的變快取決於你的環境）
- `seed`: optimizer 的 random seed
- `search_space`: **推薦**放 `SearchSpace` 物件；也可放 dict bounds

#### `objective_cfg`（「每次評估要怎麼模擬 + 怎麼算 PSD + 怎麼算 loss」）

常用鍵：

- `sim_cfg`: 每次 evaluation 的 simulation 設定
- `network_cfg`: network/whole-brain 的 connectivity + `G_net`
- `noise_cfg`: noise 設定
- `psd_cfg`: Welch PSD 設定（`nperseg`, `freq_range`, `roi`…）
- `loss_name` + `loss_cfg`（單一 loss）
- 或 `losses`（多 loss）

### 5.3 SearchSpace 怎麼設定（這決定你「優化哪些參數」）

SearchSpace 類別在 `src/wendling_sim/optimize/search_space.py`：

```python
from wendling_sim.optimize.search_space import SearchSpace

search_space = SearchSpace(
    bounds={"A": (1, 10), "B": (5, 40), "G": (5, 40), "G_net": (0, 5)},
    log_scale=set(),   # 想用 log-scale 的參數名，例如 {"a","b","g"}
    fixed={},          # 你也可以把某些參數固定，不進 optimizer
)
```

**非常重要的注意事項（會影響你怎麼寫 config）：**

- `api.optimize()` 如果收到 `opt_cfg["search_space"]` 是 **dict**，目前只會用它當 bounds：`SearchSpace(bounds=that_dict)`，`log_scale` 不會從 YAML 自動吃進來。
- 如果你真的需要 log-scale bounds（例如 a/b/g），請直接傳 `SearchSpace` 物件（見上面例子，或參考 `tests/test_optimization.py` 的寫法）。

### 5.4 `G_net` 要怎麼優化？

`api.optimize()` 會特別處理 `G_net`：

- candidate params 裡如果有 `G_net`，會被抽出來，覆蓋 `network_cfg["G_net"]`
- 其它參數會當作 `model_cfg` 傳進 simulation

所以你要優化 `G_net` 的條件是：

- 你的 SearchSpace bounds 裡要包含 `G_net`
- 你的 objective_cfg 要包含 network_cfg（或 opt_cfg.network_cfg）

### 5.5 如果你要的 objective 不是 PSD（時間域、ERP、bandpower、奇怪的統計…）

現在的 `optimize()` 是「PSD導向」：loss 只會拿到 `(psd, target_psd, freqs)`。

如果你想要用更複雜的東西當 loss（例如：
要用 time series、要看 burst event、要做某些很複雜的中間計算、要用 states），建議做法是：

1) **自己寫 objective_fn**：在裡面 `simulate()`、`compute feature`、`compute loss`
2) 用 `NevergradOptimizer` 直接跑

範例骨架（你可以直接複製改）：

```python
import numpy as np
from wendling_sim import simulate
from wendling_sim.optimize.nevergrad_engine import NevergradOptimizer
from wendling_sim.optimize.search_space import SearchSpace

def objective_fn(params):
    # 1) 你決定要不要監控 states
    res = simulate(
        sim_cfg={"dt_ms": 0.1, "duration_ms": 5000, "jit": True},
        model_cfg=params,
        network_cfg={"n_nodes": 1},  # 或 whole-brain network_cfg
        noise_cfg={"seed": 42},
        monitor_cfg={"variables": ["y0", "y1"]},  # 可選
    )

    lfp = np.asarray(res.lfp)[:, 0]

    # 2) 你自訂 feature / loss（想多複雜都可以）
    # 例如：時間域能量 + 某些頻段 bandpower 比值 + event count...
    return float(np.mean(lfp**2))

search_space = SearchSpace(bounds={"A": (1, 10), "B": (5, 40), "G": (5, 40)})
opt = NevergradOptimizer(search_space=search_space, objective_fn=objective_fn, budget=50, optimizer_name="NGOpt", seed=0)
result = opt.run(verbose=True)
print(result.best_params, result.best_loss)
```

這條路徑的優點是：你完全不受目前 `optimize()`（PSD pipeline）的限制。

---

## 6) Network / whole-brain 的完整範例（你可以照抄）

### 6.1 20 nodes toy network（generator）

```python
import numpy as np
from wendling_sim import simulate

network_cfg = {
    "n_nodes": 20,
    "generator": {"name": "erdos_renyi", "options": {"p": 0.2, "weight_dist": "lognormal", "weight_scale": 0.8, "seed": 0}},
    "normalize": "row_sum",
    "remove_self_loops": True,
    "G_net": 0.8,
}

res = simulate(
    sim_cfg={"dt_ms": 0.1, "duration_ms": 2000, "jit": True},
    model_cfg={},
    network_cfg=network_cfg,
    noise_cfg={"seed": 0, "shared": False},
)
print(res.lfp.shape)  # (T, 20)
```

### 6.2 從 HCP `.mat` structural connectivity 跑 whole-brain（demo 同款）

```python
from wendling_sim import simulate

network_cfg = {
    "W_path": "data/demo_HCP_data/102816/structural/DTI_CM.mat",
    "normalize": "row_sum",
    "G_net": 0.8,
}

# 先 load 一次，補上 n_nodes（建議）
from wendling_sim.connectivity.io import load_connectivity
conn = load_connectivity(network_cfg)
network_cfg["n_nodes"] = conn.n_nodes

res = simulate(sim_cfg={"dt_ms": 0.1, "duration_ms": 5000, "jit": True}, model_cfg={}, network_cfg=network_cfg, noise_cfg={"seed": 0})
print(res.lfp.shape)  # (T, N)
```

---

## 7) 你要「把 config 管好」：本 repo 的 config 哲學

本 repo 沒有導入 Hydra / pydantic 之類的複雜系統；核心 API 就吃 Python dict。

現有 YAML 範例在：

- `configs/sim_default.yaml`（給 `scripts/run_simulate.py` 用）
- `configs/optimize_default.yaml`（給 `scripts/run_optimize.py` 用）

建議你的實驗管理方式：

1. 每個實驗一份 YAML（或一個 Python config 檔），放在 `configs/` 或你自己的資料夾
2. 每次跑產生一個 `runs/<timestamp>/`，把：
   - `SimResult`（npz）
   - target PSD（npz）
   - best params（json）
   - history（json/csv）
   - 你自己的 plot（png）
   全部丟進去

你想要更進階的「整套 pipeline」時，最好的地方就是從 `scripts/demo_hcp_optimize.py` 開始改（它已經示範了 multi-loss、target 對齊、結果存檔/畫圖）。

---

## 8) 常見問題（FAQ / Troubleshooting）

### Q1：我不知道要在哪裡「建立 network」

你**不需要**先建立某個 Network class 才能跑。你只要提供 `network_cfg`（含 `n_nodes` + `W/W_path/generator/builder` 其一）給 `simulate()`：

- 建 network 的邏輯在 `src/wendling_sim/connectivity/io.py#load_connectivity()`
- 建 dynamical system 的邏輯在 `src/wendling_sim/sim/runner.py#run_simulation()`

### Q2：我想寫超複雜 loss（中間要算很多東西），但 optimize 只給 PSD 怎麼辦？

兩條路：

1) 還是用 `optimize()`：把你的複雜計算包成 `loss(psd, target_psd, freqs, **cfg)`（`scripts/demo_hcp_optimize.py` 就是示範）
2) 需要 time series / states：直接走「自訂 objective_fn + NevergradOptimizer」那條（見 5.5 範例）

### Q3：我改了 `sim_cfg["integrator"]` 怎麼沒差？

因為目前 `wendling_single.py` / `wendling_network.py` 的 `update()` 是手寫 Euler step，所以 DSRunner 的 integrator 設定不會改變結果（你可以視為 placeholder）。

### Q4：optimization 一直回 `inf` 或 loss 不下降

常見原因：

- 你的 loss 函式對某些輸入會炸（例如 log(0)、shape 不符）
- 你的 target PSD 頻率網格跟 simulation 不同（需要對齊或插值）
- search space 太大/太小、budget 太少
- ROI 設定不同（`roi="mean"` vs `roi="none"`）導致 shape 不一致

### Q5：`ModuleNotFoundError: No module named 'wendling_sim'`

兩種最常見的解法：

1) 進你的環境後把 repo 裝成 editable（建議）

```bash
conda activate brainpy_model
pip install -r requirements.txt
```

2) 你只是想在 repo 內快速跑：在你的 script 最上面加一段（`scripts/*.py` 就是這樣做）

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
```

### Q6：`pytest` 為什麼會 fail？

`tests/test_data.py` 目前是故意 `assert False` 的 placeholder，所以直接跑 pytest 會 fail；repo 內的 `tests/test_optimization*.py` 更像是「可執行的 demo scripts」。

---

## 9) 我想擴充功能：我應該改哪裡？

快速對照表（也可參考 `README.md` 的 map）：

| 你想改的東西 | 位置 |
|---|---|
| 單點 dynamics | `src/wendling_sim/model/wendling_single.py` |
| network coupling / 注入位置 | `src/wendling_sim/model/wendling_network.py` |
| connectivity 讀檔/正規化語意 | `src/wendling_sim/connectivity/io.py` |
| 新增 generator | `src/wendling_sim/connectivity/generators.py` |
| simulation runner（seed、monitors、輸出格式） | `src/wendling_sim/sim/runner.py` |
| PSD 內容（roi、bandpower） | `src/wendling_sim/features/psd.py` |
| 新增/改 loss | `src/wendling_sim/loss/*.py` + `src/wendling_sim/loss/registry.py`（或 runtime register） |
| optimization ask/tell loop | `src/wendling_sim/optimize/nevergrad_engine.py` |
| search space/bounds | `src/wendling_sim/optimize/search_space.py` |

---

## 10) 建議你下一步怎麼做（最有效率的學習路徑）

1. 跑一次六種 activity type，建立直覺：`conda run -n brainpy_model python scripts/run_six_types.py`
2. 先用 `roi="mean"` 做 single-node optimize（`scripts/run_optimize.py`）
3. 把 `network_cfg` 加進你的 simulate（toy network → 再換成你的 connectome）
4. 參考 `scripts/demo_hcp_optimize.py` 的 multi-loss 寫法，把你要的複雜 feature/loss 加進去
5. 如果你真的要 time-domain / states 的 objective，改走 5.5 的「自訂 objective_fn」路徑
