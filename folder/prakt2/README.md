# Dependency Visualizer — Вариант №22

**Автор:** Сигачёв Иван, ИКБО-40-24

---

## 🎯 Цель проекта

Разработать CLI-инструмент для **визуализации графа зависимостей** пакетов (аналог работы менеджера пакетов npm).
Проект реализует 5 этапов — от чтения конфигурации до генерации Mermaid-диаграммы и ASCII-дерева.
Сторонние библиотеки для получения зависимостей **не используются**.

---

## 🧩 Этап 1 — Конфигурация (CSV)

**Цель:** считать параметры из конфигурационного файла CSV и вывести их в формате `ключ=значение`.

Пример конфигурации (`config.sample.csv`):

```csv
package_name,react
repo_or_path,https://github.com/facebook/react
test_mode,false
ascii_tree,true
max_depth,2
```

**Запуск:**

```bash
python depviz.py --config config.sample.csv --stage 1
```

**Результат:**

```
package_name=react
repo_or_path=https://github.com/facebook/react
test_mode=False
ascii_tree=True
max_depth=2
```

✅ Реализована проверка и обработка ошибок для каждого параметра.

---

## ⚙️ Этап 2 — Сбор данных

**Цель:** получить **прямые зависимости** пакета.
Поддерживаются два режима:

* `test_mode=true` — тестовый CSV-граф;
* `test_mode=false` — чтение `package.json` (локального или из репозитория).

### 🔹 Тестовый пример

```bash
python depviz.py --config examples/config_graph1.csv --stage 2
```

**Результат:**

```
B
C
```

### 🔹 Реальный пример (локальный `react_package.json`)

```bash
python depviz.py --config config.local.csv --stage 2
```

**Результат:**

```
loose-envify
object-assign
react-dom
scheduler
```

---

## 🕸 Этап 3 — Построение графа зависимостей

**Цель:** построить граф зависимостей с помощью BFS,
учитывая транзитивность и максимальную глубину.

**Команда:**

```bash
python depviz.py --config examples/config_graph2.csv --stage 3
```

**Результат:**

```
NODES: A, B, C, D, E
EDGES:
A -> B
B -> C
B -> D
C -> A
D -> E
```

✅ Обрабатываются циклические зависимости (`A ↔ C`).
✅ Реализован контроль глубины обхода.

---

## 🔄 Этап 4 — Обратные зависимости

**Цель:** вывести пакеты, которые **зависят от** заданного.

**Команда:**

```bash
python depviz.py --config examples/config_graph2.csv --stage 4 --reverse-of A
```

**Результат:**

```
REVERSE DEPENDENCIES (who depends on A):
A <- C
B <- A
C <- B
```

✅ Используется тот же алгоритм обхода (BFS) с реверсом направлений рёбер.

---

## 🧭 Этап 5 — Визуализация

**Цель:** вывести граф зависимостей в текстовом виде:

* **Mermaid** — для построения диаграмм.
* **ASCII-дерево** — для терминала (если включен `ascii_tree=true`).

**Команда:**

```bash
python depviz.py --config examples/config_graph1.csv --stage 5
```

**Результат:**

```
graph TD
  A --> B
  A --> C
  B --> D
  C --> D
  C --> E

--- ASCII TREE ---
A
├─ B
├─ └─ D
└─ C
└─ ├─ D
└─ └─ E
```

🎨 Mermaid-диаграмму можно вставить на сайт [https://mermaid.live](https://mermaid.live).

---

## 🧰 Локальное тестирование без интернета

Для оффлайн-демонстрации используется локальный `package.json` и конфигурации:

### `react_package.json`

```json
{
  "name": "react",
  "version": "18.3.1",
  "dependencies": {
    "loose-envify": "^1.4.0",
    "object-assign": "^4.1.1",
    "scheduler": "^0.23.0"
  },
  "peerDependencies": {
    "react-dom": "^18.0.0"
  }
}
```

### `config.local.csv`

```csv
package_name,react
repo_or_path,react_package.json
test_mode,false
ascii_tree,false
max_depth,1
```

### Проверка:

```bash
python depviz.py --config config.local.csv --stage 2
```

---

## 📂 Структура проекта

```
depviz_variant22/
├── depviz.py
├── react_package.json
├── config.local.csv
├── config.local_tree.csv
├── examples/
│   ├── config_graph1.csv
│   └── config_graph2.csv
└── README.md
```

---

## 🧾 Что показать на защите

| Этап | Что демонстрировать                  | Пример вывода |
| ---- | ------------------------------------ | ------------- |
| 1    | Вывод параметров из CSV              | ✅             |
| 2    | Прямые зависимости (тест и локально) | ✅             |
| 3    | Построение графа и цикл              | ✅             |
| 4    | Обратные зависимости                 | ✅             |
| 5    | Mermaid + ASCII-дерево               | ✅             |

---

## ✅ Итог

Проект **depviz_variant22** полностью реализует требования Варианта №22:

* CLI-интерфейс и конфигурация CSV
* анализ зависимостей npm
* обработка ошибок и циклов
* визуализация в формате Mermaid и ASCII

**Готово к защите и загрузке на GitHub.**
