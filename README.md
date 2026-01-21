# LLM Safety Alignment через Mechanistic Interpretability

## Обзор

Проект исследует интерпретируемость LLM для задач безопасности через **Sparse Autoencoder (SAE)**.

Pipeline:
1. Извлечение внутренних активаций выбранного слоя LLM
2. Прогон активаций через SAE → получение **sparse-кодов** (features)
3. Обучение бинарного классификатора на признаках SAE для детекта вредного контента
4. Сравнение с baseline: Dense Linear Probe на сырых активациях

> Важно: в анализе ниже под `X` подразумеваются **коды SAE** (выход энкодера после ReLU/threshold), а не реконструкция и не входные активации LLM.

---

## Быстрый старт

### Системные требования
Минимум для инференса на GPU:
- RAM: 12 GB
- VRAM: 6 GB

### Установка
```bash
pip install torch transformers numpy pandas matplotlib scikit-learn tqdm
```

### Запуск анализа активаций
1. Убедитесь, что у вас есть папка с чанками (например, `sae_feats_layer20/`)
2. Откройте ноутбук `feature_analysis.ipynb`
3. В первой ячейке настройте `DUMP_DIR` на путь к папке с активациями
4. Запустите все ячейки

---

## Генерация и сохранение графиков

При запуске `feature_analysis.ipynb` изображения автоматически сохраняются в `activation_report/`.

Настройки сохранения:
- тёмный фон,
- `bbox_inches="tight"`,
- `dpi=200`.

Для просмотра сохранённых изображений в ноутбуке используйте:
```python
show_img("01_firing_rate_hist.png", width=900)
```

---

## Датасеты

### Датасеты для обучения SAE

SAE обучался на смеси датасетов:

**Общие:**
- ShareGPT — https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
- openwebtext-10k — https://huggingface.co/datasets/stas/openwebtext-10k

**Специфичные для безопасности:**
- AdvBench — https://huggingface.co/datasets/walledai/AdvBench
- ChatGPT-Jailbreak — https://huggingface.co/datasets/rubend18/ChatGPT-Jailbreak-Prompts
- hh-rlhf — https://huggingface.co/datasets/Anthropic/hh-rlhf

**Распределение по примерам:**
- unsafe: 9899 (12.49%)
- sharegpt: 59369 (74.93%)
- openwebtext: 9887 (12.48%)
- jlbrk: 79 (0.10%)

**Распределение по токенам:**
- unsafe: 4,210,852 (4.21%)
- sharegpt: 86,138,752 (86.14%)
- openwebtext: 9,613,602 (9.61%)
- jlbrk: 35,169 (0.04%)

### Датасет для анализа активаций

Для сбора активаций использовался **Galtea Red Teaming: Non-Commercial Subset**:
- https://huggingface.co/datasets/Galtea-AI/galtea-red-teaming-clustered-data

Кластеры (семантические категории вредного поведения):
- Cluster 0: Ambiguous Requests
- Cluster 1: Jailbreak & Roleplay Attacks
- Cluster 2: Financial Stuff & Fraud
- Cluster 3: Toxicity & Hate Speech
- Cluster 4: Violence & Illegal Activities
- Cluster 5: Privacy Violations

---

## Анализ SAE-активаций

### Графики

Все изображения лежат в папке [`activation_report/`](activation_report/).

<p>
  <img src="activation_report/01_firing_rate_hist.png" width="380" />
  <img src="activation_report/02_log10_firing_rate_hist.png" width="380" />
</p>
<p>
  <img src="activation_report/03_top30_firing_rate.png" width="380" />
  <img src="activation_report/04_top30_max_activation.png" width="380" />
</p>
<p>
  <img src="activation_report/05_top30_mean_when_active.png" width="380" />
  <img src="activation_report/06_active_features_per_token_hist.png" width="380" />
</p>
<p>
  <img src="activation_report/07_pca_scatter.png" width="380" />
</p>

Список файлов:
- `01_firing_rate_hist.png`
- `02_log10_firing_rate_hist.png`
- `03_top30_firing_rate.png`
- `04_top30_max_activation.png`
- `05_top30_mean_when_active.png`
- `06_active_features_per_token_hist.png`
- `07_pca_scatter.png`

---

## Как читать эти графики

Ниже — **что измеряет каждый график**, и **какие паттерны** обычно говорят о проблемах со sparsity.

### 1) Firing rate distribution (01)

**Определение:** для каждой фичи \(j\) firing rate
\[
fr_j = \mathbb{E}[\mathbb{1}(X_{:,j} > \varepsilon)]
\]
где \(X\) — коды SAE (tokens × features), \(\varepsilon\) — порог активности.

Что обычно ожидается у SAE:
- большая доля фич имеет очень маленький \(fr\) (редко активируются),
- небольшая доля фич активируется заметно чаще.

Красные флаги:
- выраженный пик около **1.0** → много *always-on* фич (почти всегда \(X>0\)).

### 2) log10(firing rate) (02)

Используется для \(fr>0\), чтобы “растянуть” редкие события.
- Если значимая масса точек у **\(fr\approx 1\)** (т.е. \(\log_{10}(fr)\approx 0\)), это подтверждает наличие *always-on* фич.

### 3) Top-30 firing rate (03)

Если top-30 имеют \(fr\) близко к 1 (например, 0.999…), это не “шум” — это реально почти постоянная активность этих фич.

### 4–5) Max activation и Mean when active (04–05)

Эти графики отвечают на вопросы:
- насколько “высоко” могут активироваться фичи (max),
- насколько велики значения **условно на активных позициях** (mean_when_active).

Сами по себе большие max/mean не всегда проблема. Проблема — когда они сочетаются с *always-on*:
- это часто означает сильный **bias/offset** или несоответствие **масштаба** входов SAE.

### 6) Active features per token (06)

Считает \(L_0\) по токенам:
\[
L_0(t) = \sum_j \mathbb{1}(X_{t,j} > \varepsilon)
\]

Для SAE обычно ожидается, что активных фич на токен **значительно меньше**, чем размерность словаря (в разы и на порядки).

Красный флаг:
- пик на значениях, близких к половине словаря или выше → коды **плотные**, а не sparse.

### 7) PCA scatter (07)

PCA — это быстрый “smoke test”:
- если видны “режимы” (например, полоса + отдельное облако), это может быть:
  - смесь разных источников/чанков с разной нормировкой,
  - доминирование смещения/масштаба (offset/scale),
  - спец-токены/артефакты препроцессинга.

Практика:
- делайте PCA на **центрированных** признаках, а лучше на стандартизованных (mean+std).
- если структура резко меняется после центрирования — проблема была в offset/scale.

---

## Что показал текущий прогон (observed)

Анализ выполнен на подвыборке из **13,676 токенов**.

**Форма тензора:**
- `X.shape = [13676, 20480]` — 13,676 токенов × 20,480 фич

**Статистика значений:**
- min = 0.0  
- median = 0.234375  
- mean = 2.2336  
- max = 780.0  
- percentiles: p1=0, p5=0, p50=0.234375, p95=2.25, p99=55.5

**Доля активных значений при разных порогах:**

| eps | frac(X > eps) | active_per_token_mean |
|-----|---------------|----------------------|
| 0.0   | 0.5793 | 11863.6 |
| 1e-4  | 0.5793 | 11863.3 |
| 1e-3  | 0.5790 | 11856.9 |
| 1e-2  | 0.5760 | 11796.2 |
| 0.1   | 0.5459 | 11180.0 |

**L0 по токенам** (`L0 = (X > 0).sum(dim=1)`):
- mean = 11863.6
- median = 12123.0
- p95 = 12819.0
- p99 = 13062.0

### Главный вывод: коды не sparse

Сейчас коды SAE выглядят **плотными**:
- активны ~57.9% элементов матрицы,
- в среднем активны ~11.9k из 20.5k фич на токен (≈58%).

Это сильно выше типичного уровня sparsity, который ожидают от SAE (как минимум на порядки меньше активных фич на токен).

---

## Наиболее вероятные причины (и что проверить)

Ниже — приоритетный чек-лист, который помогает отличить “SAE реально плохо обучился” от “мы анализируем не то / не так”.

### 1) Bias/offset → много always-on
Если у энкодера большой положительный bias **или** входы SAE имеют сильный сдвиг, множество фич становятся стабильно положительными.

Что проверить:
- долю точных нулей: `frac(X == 0)`
- распределение значений вблизи 0 (например, гистограмма на [0, 0.5])

### 2) Несовпадение нормировок train vs inference
Если SAE обучался на центрированных/нормированных активациях, а анализ делается на “сыром” масштабе, sparsity может разрушиться.

### 3) Слабая sparsity-регуляризация / нет жёсткого разреживания

---

## Структура проекта

```text
security_sae/
├── feature_analysis.ipynb      # Основной ноутбук для анализа активаций
├── get_activations.py          # Извлечение SAE-активаций из датасета
├── analyze_activations.py      # Быстрый анализ статистики
├── dataset_viz.py              # Визуализация датасета
├── activation_report/          # Сохранённые графики анализа
│   ├── 01_firing_rate_hist.png
│   ├── 02_log10_firing_rate_hist.png
│   ├── 03_top30_firing_rate.png
│   ├── 04_top30_max_activation.png
│   ├── 05_top30_mean_when_active.png
│   ├── 06_active_features_per_token_hist.png
│   └── 07_pca_scatter.png
├── sae_feats_layer20/          # Чанки с кодами SAE
│   ├── meta.json
│   ├── sae_feats_layer20_chunk*.pt
│   └── feature_stats.npz
├── activations/                # Чекпоинты SAE
│   └── ckpt_epoch_15.pt
├── loaders/
│   ├── main.py
│   ├── sharegpt.py
│   ├── openwebtext.py
│   ├── jailbreak.py
│   ├── advbench.py
│   ├── hh_rlhf.py
│   └── dataset_mixer.py
└── dataset/                    # Обработанные датасеты
```

---

## Использование скриптов

### Извлечение активаций
```bash
python get_activations.py
```
Скрипт читает датасет из CSV, извлекает активации слоя LLM, прогоняет их через SAE и сохраняет коды в чанки.

### Быстрый анализ статистики
```bash
python analyze_activations.py --dump_dir ./sae_feats_layer20 --sample_tokens 20000
```
Скрипт вычисляет статистику по токенам и сохраняет `feature_stats.npz`, а также печатает отчёт по sparsity.
