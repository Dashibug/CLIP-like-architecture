# Определение страны документа по фото (ViT + текстовые прототипы)

> ViT извлекает признаки изображения, текстовый энкодер формирует прототипы стран. На инференсе сравниваем эмбеддинг изображения с матрицей прототипов и берём `top-k`.


<img width="522" height="492" alt="image" src="https://github.com/user-attachments/assets/3ada7639-015c-4e18-b065-4fce0ff41110" />

---

## Идея и метод

- Используется визуальный энкодер **ViT** (`timm`) и многоязычный текстовый энкодер **SentenceTransformer**.  
- Для каждой страны строятся **текстовые прототипы** — набор промптов на английском и локальных языках (например, *España*, *Deutschland*, *Türkiye* и т.д.). Эти промпты кодируются текстовым энкодером и **усредняются → один вектор на страну**.  
- Модель обучается **CLIP-подобной** контрастивной функцией потерь: сближаем эмбеддинг изображения и эмбеддинг текста.  
- **Инференс без текста**: достаточно ViT-эмбеддинга изображения и **матрицы прототипов** (скалярные произведения + выбор `top-k`).  

---

## Плюсы подхода

- **Масштабируемость:** чтобы добавить страну, достаточно дописать название и пересчитать прототипы (переобучение не требуется).  
- **Стабильность к формулировкам:** используем набор промптов и усреднение.  
- **Обобщение по типам документов:** в шаблонах предусмотрены *passport / ID card / driver license / voter card* и т.д.  
- **Быстрый инференс:** только ViT + матрица прототипов.  

---

## Что внутри ноутбука

- Установка зависимостей  
- Конфиг  
- Архитектура и прототипы  
- Обучение и валидация  
- Инференс  
- Анализ ошибок  
- Ручная проверка на картинке  

### Ключевые компоненты
- `CountryVLM` — ViT-бэкбон + две линейные проекции (для изображения и текста) + обучаемая температура.  
- `FrozenTextEncoder` — многоязычный `SentenceTransformer` (замороженный).  
- `CountryPrototypes` — контейнер для текстовых прототипов.  
- `CountryDataset` — датасет из CSV с колонками `path, iso3`.  
- Тренировочный цикл: **cosine LR**, **AMP**, `clip_grad_norm`, чекпойнты по эпохам и сохранение **лучшей** модели и прототипов.  

---

## Данные

Ожидаются CSV-файлы с колонками:

```csv
path,iso3
dataset/GBR/GBR_55.jpg,GBR
dataset/ESP/ESP_12.jpg,ESP
...
```

- `path` — путь к изображению  
- `iso3` — код страны в формате ISO-3

Структура папок может быть любой — важен корректный путь в CSV.

---

## Поддерживаемые страны

<details>
<summary>Показать список (ISO3 — название)</summary>

ARG — Argentina • AUS — Australia • BEL — Belgium • BGR — Bulgaria • BLR — Republic of Belarus • BRA — Brazil • CAN — Canada • CHL — Chile • CHN — China • DEU — Germany • DOM — Dominican Republic • ESP — Spain • EST — Estonia • FRA — France • GBR — United Kingdom • HUN — Hungary • IDN — Indonesia • IND — India • IRL — Ireland • ITA — Italy • JPN — Japan • KAZ — Kazakhstan • KGZ — Kyrgyzstan • KOR — South Korea • MDA — Moldova • MEX — Mexico • NLD — Netherlands • POL — Poland • RUS — Russia • SVK — Slovakia • SWE — Sweden • TUR — Turkey • UKR — Ukraine • USA — United States • UZB — Uzbekistan • ZAF — South Africa

</details>

---
