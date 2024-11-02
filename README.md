# Demand Forecasting and Stock Management System

Проект посвящен построению системы управления запасами товаров. Целью является автоматизация прогнозирования спроса для снижения затрат, связанных с нехваткой или избытком товара, и повышения оборачиваемости склада.

## О проекте

Система управления запасами включает:

- **Прогнозирование спроса** с использованием квантильной регрессии для оценки спроса по различным уровням уверенности (пессимистичный, консервативный, оптимистичный прогнозы).
- **Поддержка автоматизации закупок** с помощью FastAPI-сервиса, который предоставляет рекомендации по пополнению складских запасов.
- **MLOps с ClearML**: создание, отслеживание и автоматизация пайплайнов для подготовки данных, обучения моделей и прогнозирования спроса.

## Основные компоненты

### 1. Датасет и подготовка данных

Система работает с данными о продажах. Основные операции включают:

- **Агрегацию продаж** — суммирование проданных единиц товара (qty) по дням.
- **Генерацию признаков** с помощью оконных функций, таких как скользящее среднее и квантиль продаж за последние 7, 14 и 21 день.
- **Создание целевых переменных** — суммарных продаж за следующие 7, 14 и 21 день.

### 2. Модель

Проект включает обучающую модель **MultiTargetModel**, которая нацелена на квантильную регрессию:

- Прогнозы генерируются для каждого SKU по трём горизонтам (7, 14 и 21 день) и для квантилей 0.1, 0.5 и 0.9.
- Модель обучается отдельно для каждого SKU, чтобы лучше учитывать уникальные особенности спроса на товары.

#### Метрика
Функция потерь **quantile loss** позволяет учитывать асимметричность потерь для различных квантилей:
- 0.1 квантиль — "пессимистичный" прогноз, наказывает за перепредсказание.
- 0.5 квантиль — "консервативный" прогноз, симметричен.
- 0.9 квантиль — "оптимистичный" прогноз, наказывает за недопредсказание.

### 3. Пайплайны

Система использует два пайплайна, созданных с помощью ClearML:

1. **Training Pipeline** — обучение модели на основе последних доступных данных, сохранение обученной модели в ClearML Model Registry.
2. **Inference Pipeline** — получение данных за последние 21 день, генерация прогнозов и сохранение предсказаний для использования в FastAPI сервисе.

### 4. FastAPI-сервис

Сервис предоставляет REST API для управления запасами:

- **upload_predictions** — загрузка файла с прогнозами.
- **how_much_to_order** — возвращает рекомендованное количество для заказа, чтобы покрыть прогнозируемый спрос.
- **stock_level_forecast** — рассчитывает остаток на складе для заданного горизонта и уровня уверенности.
- **low_stock_sku_list** — возвращает список SKU, запасы которых ниже уровня прогнозируемого спроса.

### Документация API (Swagger)

FastAPI предоставляет встроенную поддержку Swagger-документации, позволяя просматривать и тестировать все доступные конечные точки (эндпоинты) API:

- После запуска сервиса Swagger UI доступен по адресу: http://localhost:5000/docs.
- Swagger автоматически отображает описание всех доступных API методов, их параметры и возвращаемые значения, а также позволяет выполнять тестовые запросы непосредственно из интерфейса.

## Запуск

### 1. Настройка ClearML

Для использования ClearML установите его и настройте доступ:
```bash
pip install clearml
clearml-init
```

### 2. Запуск FastAPI

Запустите FastAPI сервис:
```bash
python app.py
```

### 3. Выполнение пайплайнов

#### Training Pipeline
Запустите тренировочный пайплайн для обучения модели:
```bash
python pipelines/training.py
```

#### Inference Pipeline
Запустите инференс пайплайн для генерации прогнозов:
```bash
python pipelines/inference.py
```