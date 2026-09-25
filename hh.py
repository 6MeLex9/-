# -*- coding: utf-8 -*-
"""
Прогнозирование типа смартфона (Айфон / Андроид)
методом k ближайших соседей (KNeighborsClassifier)

Возможности:
  * обучение модели с подбором гиперпараметров;
  * кросс-валидация;
  * РЕЖИМ ДЕМОНСТРАЦИИ — ручной ввод значений нового пользователя;
  * РЕЖИМ ТЕСТИРОВАНИЯ — прогон по заранее заданным профилям
    с известным правильным ответом.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, MultiLabelBinarizer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------
# 0. ПУТИ И КОНСТАНТЫ
# ------------------------------------------------------------------
FILE = "Предпочтение пользователей Iphone или android.csv"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# ------------------------------------------------------------------
# 1. ЗАГРУЗКА ДАННЫХ
# ------------------------------------------------------------------
df = pd.read_csv(FILE)

print("=" * 70)
print("РАЗМЕР ТАБЛИЦЫ:", df.shape)

target_col = df.columns[-1]
print("ЦЕЛЕВАЯ ПЕРЕМЕННАЯ:", repr(target_col))
print(df[target_col].value_counts())

drop_cols = [target_col, df.columns[0]]
X = df.drop(columns=drop_cols).copy()
y = df[target_col].astype(str).str.strip()
X.columns = [c.strip() for c in X.columns]

# ------------------------------------------------------------------
# 2. УДАЛЯЕМ ШУТОЧНЫЙ ПРИЗНАК "ЯБЛОКИ"
# ------------------------------------------------------------------
apple_col = "21. Как часто вы кушаете яблоки от 1 до 10?"
if apple_col in X.columns:
    X = X.drop(columns=[apple_col])

# ------------------------------------------------------------------
# 3. ОЧИСТКА ЧИСЛОВЫХ ЗНАЧЕНИЙ
# ------------------------------------------------------------------
def clean_numeric(series: pd.Series) -> pd.Series:
    s = (series.astype(str)
              .str.replace("\u00a0", "", regex=False)
              .str.replace(" ", "", regex=False)
              .str.replace(".", "", regex=False)
              .str.replace(",", ".", regex=False))
    return pd.to_numeric(s, errors="coerce")

# ------------------------------------------------------------------
# 4. ЧИСЛОВЫЕ И КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ
# ------------------------------------------------------------------
numeric_candidates = [
    "7.Сколько Вам лет?",
    "8. Сколько в среднем вы готовы потратить на новый телефон?",
    "9. Насколько вам важна возможность кастомизации интерфейса?",
    "13. Как часто вы меняете телефон? В годах.",
    "18. Сколько раз в день вы примерно заряжаете телефон? В ответ число",
]
numeric_cols = [c for c in numeric_candidates if c in X.columns]

for col in numeric_cols:
    X[col] = clean_numeric(X[col])

categorical_cols = [c for c in X.columns if c not in numeric_cols]

# ------------------------------------------------------------------
# 5. РАЗБИВАЕМ МУЛЬТИВЫБОР (столбец с ';')
# ------------------------------------------------------------------
multi_col = "4. За что вы готовы переплатить, покупая смартфон?"
multi_categories = []
if multi_col in X.columns:
    mlb = MultiLabelBinarizer()
    multi_values = X[multi_col].fillna("").astype(str).str.split(";")
    multi_enc = pd.DataFrame(
        mlb.fit_transform(multi_values),
        columns=[f"perплата_{c.strip()}" for c in mlb.classes_],
        index=X.index
    )
    multi_categories = [c.strip() for c in mlb.classes_]
    X = pd.concat([X.drop(columns=[multi_col]), multi_enc], axis=1)
    categorical_cols = [c for c in categorical_cols if c != multi_col]
    categorical_cols.extend(multi_enc.columns.tolist())

# ------------------------------------------------------------------
# 6. ПАЙПЛАЙН ПРЕДОБРАБОТКИ
# ------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

# ------------------------------------------------------------------
# 7. TRAIN / TEST
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

# ------------------------------------------------------------------
# 8. ПОДБОР k
# ------------------------------------------------------------------
knn_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("knn", KNeighborsClassifier())
])

max_k = max(3, min(11, len(X_train) - 1))
odd_ks = list(range(3, max_k + 1, 2))

param_grid = {
    "knn__n_neighbors": odd_ks,
    "knn__weights":     ["uniform", "distance"],
    "knn__metric":      ["euclidean", "manhattan"],
}
grid = GridSearchCV(
    knn_pipeline, param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring="accuracy", n_jobs=-1
)
grid.fit(X_train, y_train)

print("=" * 70)
print("ЛУЧШИЕ ПАРАМЕТРЫ:", grid.best_params_)
print(f"ЛУЧШАЯ CV-ТОЧНОСТЬ: {grid.best_score_:.3f}")

best_model = grid.best_estimator_

# ------------------------------------------------------------------
# 9. ОЦЕНКА
# ------------------------------------------------------------------
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("=" * 70)
print(f"ACCURACY НА ТЕСТЕ: {acc:.3f}")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=best_model.classes_).plot(
    cmap="Blues", ax=ax, colorbar=False
)
ax.set_title("Матрица ошибок KNN")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring="accuracy")
print("=" * 70)
print(f"CV ACCURACY: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ==================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ИНТЕРАКТИВА
# ==================================================================

# Шаблон ответов по умолчанию (используется в демо-режиме)
DEFAULT_PROFILE = {
    "1. Стабильность или новшество?": "Стабильность",
    "2. Часто ли вы фотографируете?": "Каждый день",
    "3.Ведете ли вы блог?": "Нет",
    multi_col: "Экосистема;Надежность",
    "5. Статус или технологичность?": "Технологичность",
    '6. VPN или "запрет"?': "VPN",
    "7.Сколько Вам лет?": 22,
    "8. Сколько в среднем вы готовы потратить на новый телефон?": 70000,
    "9. Насколько вам важна возможность кастомизации интерфейса?": 5,
    "10.Можете ли вы поставить рингтон из файла?": "Да",
    "11.Как вы относитесь к перепродаже своего смартфона спустя 2-3 года?": "Хорошо",
    "12. Насколько Вам важна возможность настройки системы под себя?": "Да",
    "13. Как часто вы меняете телефон? В годах.": 3,
    "14. Насколько важна актуальность и обслуживание старых версий ОС?": "Важна",
    "15. Ваша сфера деятельности?": "IT",
    "16. Какого вы пола?": "М",
    "17. Вы ведете инстаграм?": "Нет",
    "18. Сколько раз в день вы примерно заряжаете телефон? В ответ число": 2,
    "19. Что бы вы выбрали экономить и откладывать или порадовать себя?": "Порадовать себя",
    "20. Ваша гарнитура от одного бренда? или от разных?": "От одного бренда",
}


def profile_to_dataframe(profile: dict) -> pd.DataFrame:
    """Преобразует словарь-профиль в DataFrame, согласованный с X."""
    row = pd.DataFrame([profile]).reindex(columns=X.columns)

    # Числовые
    for col in numeric_cols:
        if col in row.columns:
            row[col] = clean_numeric(row[col])

    # Мультивыбор
    for col in X.columns:
        if col.startswith("perплата_"):
            cat = col.replace("perплата_", "")
            raw = str(profile.get(multi_col, ""))
            row[col] = int(cat in [s.strip() for s in raw.split(";")])

    return row.fillna(0)


def show_available_values(field: str):
    """Показывает варианты ответа для категориального поля."""
    if field in multi_categories and field == multi_col:
        print(f"   Варианты (через ';'): {', '.join(multi_categories)}")
    elif field in X.columns and field not in numeric_cols:
        vals = sorted(df[field].dropna().astype(str).unique()) if field in df.columns else []
        if vals:
            print(f"   Возможные ответы: {', '.join(vals)}")


def ask_user_profile() -> dict:
    """Интерактивный ввод профиля (Enter — оставить значение по умолчанию)."""
    print("\n--- ВВОД ПРОФИЛЯ НОВОГО ПОЛЬЗОВАТЕЛЯ ---")
    print("Нажмите Enter, чтобы оставить значение по умолчанию.")
    print("Введите '?' для просмотра возможных ответов.\n")

    profile = {}
    for i, field in enumerate(DEFAULT_PROFILE.keys(), 1):
        default = DEFAULT_PROFILE[field]
        while True:
            raw = input(f"{i:>2}. {field}\n    [{default}]: ").strip()
            if raw == "":
                profile[field] = default
                break
            if raw == "?":
                show_available_values(field)
                continue
            # Приведение типа для числовых полей
            if field in numeric_cols:
                try:
                    profile[field] = float(raw) if "." in raw or "," in raw else int(raw)
                    break
                except ValueError:
                    print("   ⚠ Введите число.")
                    continue
            profile[field] = raw
            break
    return profile


def predict_profile(profile: dict):
    """Возвращает (класс, вероятность, все вероятности по классам)."""
    row = profile_to_dataframe(profile)
    pred = best_model.predict(row)[0]
    proba = best_model.predict_proba(row)[0]
    classes = best_model.classes_
    proba_dict = dict(zip(classes, proba))
    return pred, max(proba), proba_dict


# ==================================================================
# ЗАРАНЕЕ ЗАДАННЫЕ ПРОФИЛИ ДЛЯ ТЕСТИРОВАНИЯ
# ==================================================================
TEST_PROFILES = [
    {
        "name": "IT-специалист, лоялен к экосистеме Apple",
        "expected": "Айфон",
        "profile": {
            "1. Стабильность или новшество?": "Стабильность",
            "2. Часто ли вы фотографируете?": "Каждый день",
            "3.Ведете ли вы блог?": "Нет",
            multi_col: "Экосистема;Надежность",
            "5. Статус или технологичность?": "Технологичность",
            '6. VPN или "запрет"?': "VPN",
            "7.Сколько Вам лет?": 22,
            "8. Сколько в среднем вы готовы потратить на новый телефон?": 80000,
            "9. Насколько вам важна возможность кастомизации интерфейса?": 1,
            "10.Можете ли вы поставить рингтон из файла?": "Нет",
            "11.Как вы относитесь к перепродаже своего смартфона спустя 2-3 года?": "Хорошо",
            "12. Насколько Вам важна возможность настройки системы под себя?": "Не знаю",
            "13. Как часто вы меняете телефон? В годах.": 3,
            "14. Насколько важна актуальность и обслуживание старых версий ОС?": "Важна",
            "15. Ваша сфера деятельности?": "IT",
            "16. Какого вы пола?": "М",
            "17. Вы ведете инстаграм?": "Нет",
            "18. Сколько раз в день вы примерно заряжаете телефон? В ответ число": 2,
            "19. Что бы вы выбрали экономить и откладывать или порадовать себя?": "Порадовать себя",
            "20. Ваша гарнитура от одного бренда? или от разных?": "От одного бренда",
        },
    },
    {
        "name": "Студент, экономный, любит кастомизацию",
        "expected": "Андроид",
        "profile": {
            "1. Стабильность или новшество?": "Стабильность",
            "2. Часто ли вы фотографируете?": "Раз в месяц",
            "3.Ведете ли вы блог?": "Нет",
            multi_col: "Надежность",
            "5. Статус или технологичность?": "Технологичность",
            '6. VPN или "запрет"?': "VPN",
            "7.Сколько Вам лет?": 21,
            "8. Сколько в среднем вы готовы потратить на новый телефон?": 25000,
            "9. Насколько вам важна возможность кастомизации интерфейса?": 7,
            "10.Можете ли вы поставить рингтон из файла?": "Да",
            "11.Как вы относитесь к перепродаже своего смартфона спустя 2-3 года?": "Плохо",
            "12. Насколько Вам важна возможность настройки системы под себя?": "Да",
            "13. Как часто вы меняете телефон? В годах.": 5,
            "14. Насколько важна актуальность и обслуживание старых версий ОС?": "Важна",
            "15. Ваша сфера деятельности?": "Студент",
            "16. Какого вы пола?": "М",
            "17. Вы ведете инстаграм?": "Нет",
            "18. Сколько раз в день вы примерно заряжаете телефон? В ответ число": 1,
            "19. Что бы вы выбрали экономить и откладывать или порадовать себя?": "Экономить и откладывать",
            "20. Ваша гарнитура от одного бренда? или от разных?": "От разных брендов",
        },
    },
    {
        "name": "Бизнес-леди, статус превыше всего",
        "expected": "Айфон",
        "profile": {
            "1. Стабильность или новшество?": "Новшество",
            "2. Часто ли вы фотографируете?": "Каждый день",
            "3.Ведете ли вы блог?": "Да",
            multi_col: "Статус",
            "5. Статус или технологичность?": "Статус",
            '6. VPN или "запрет"?': "VPN",
            "7.Сколько Вам лет?": 34,
            "8. Сколько в среднем вы готовы потратить на новый телефон?": 100000,
            "9. Насколько вам важна возможность кастомизации интерфейса?": 10,
            "10.Можете ли вы поставить рингтон из файла?": "Да",
            "11.Как вы относитесь к перепродаже своего смартфона спустя 2-3 года?": "Хорошо",
            "12. Насколько Вам важна возможность настройки системы под себя?": "Не знаю",
            "13. Как часто вы меняете телефон? В годах.": 3,
            "14. Насколько важна актуальность и обслуживание старых версий ОС?": "Не важна",
            "15. Ваша сфера деятельности?": "Бизнес",
            "16. Какого вы пола?": "Ж",
            "17. Вы ведете инстаграм?": "Да",
            "18. Сколько раз в день вы примерно заряжаете телефон? В ответ число": 3,
            "19. Что бы вы выбрали экономить и откладывать или порадовать себя?": "Порадовать себя",
            "20. Ваша гарнитура от одного бренда? или от разных?": "От одного бренда",
        },
    },
    {
        "name": "Дизайнер, любит эксперименты и Android",
        "expected": "Андроид",
        "profile": {
            "1. Стабильность или новшество?": "Новшество",
            "2. Часто ли вы фотографируете?": "Каждый день",
            "3.Ведете ли вы блог?": "Да",
            multi_col: "Камера",
            "5. Статус или технологичность?": "Технологичность",
            '6. VPN или "запрет"?': "VPN",
            "7.Сколько Вам лет?": 22,
            "8. Сколько в среднем вы готовы потратить на новый телефон?": 50000,
            "9. Насколько вам важна возможность кастомизации интерфейса?": 10,
            "10.Можете ли вы поставить рингтон из файла?": "Да",
            "11.Как вы относитесь к перепродаже своего смартфона спустя 2-3 года?": "Хорошо",
            "12. Насколько Вам важна возможность настройки системы под себя?": "Да",
            "13. Как часто вы меняете телефон? В годах.": 2,
            "14. Насколько важна актуальность и обслуживание старых версий ОС?": "Важна",
            "15. Ваша сфера деятельности?": "Дизайн",
            "16. Какого вы пола?": "Ж",
            "17. Вы ведете инстаграм?": "Да",
            "18. Сколько раз в день вы примерно заряжаете телефон? В ответ число": 2,
            "19. Что бы вы выбрали экономить и откладывать или порадовать себя?": "Экономить и откладывать",
            "20. Ваша гарнитура от одного бренда? или от разных?": "От одного бренда",
        },
    },
    {
        "name": "Программист, ценит надёжность и стабильность",
        "expected": "Андроид",
        "profile": {
            "1. Стабильность или новшество?": "Стабильность",
            "2. Часто ли вы фотографируете?": "Раз в неделю",
            "3.Ведете ли вы блог?": "Нет",
            multi_col: "Экосистема;Надежность",
            "5. Статус или технологичность?": "Технологичность",
            '6. VPN или "запрет"?': "VPN",
            "7.Сколько Вам лет?": 22,
            "8. Сколько в среднем вы готовы потратить на новый телефон?": 60000,
            "9. Насколько вам важна возможность кастомизации интерфейса?": 2,
            "10.Можете ли вы поставить рингтон из файла?": "Да",
            "11.Как вы относитесь к перепродаже своего смартфона спустя 2-3 года?": "Хорошо",
            "12. Насколько Вам важна возможность настройки системы под себя?": "Да",
            "13. Как часто вы меняете телефон? В годах.": 4,
            "14. Насколько важна актуальность и обслуживание старых версий ОС?": "Важна",
            "15. Ваша сфера деятельности?": "Программирование",
            "16. Какого вы пола?": "М",
            "17. Вы ведете инстаграм?": "Нет",
            "18. Сколько раз в день вы примерно заряжаете телефон? В ответ число": 1,
            "19. Что бы вы выбрали экономить и откладывать или порадовать себя?": "Экономить и откладывать",
            "20. Ваша гарнитура от одного бренда? или от разных?": "От одного бренда",
        },
    },
]


def run_demo_mode():
    """Ручной ввод профиля и прогноз."""
    profile = ask_user_profile()
    pred, prob, all_proba = predict_profile(profile)

    print("\n" + "-" * 60)
    print("РЕЗУЛЬТАТ ПРОГНОЗА")
    print(f"  Тип телефона: {pred}")
    print(f"  Уверенность:  {prob:.2%}")
    print("  Распределение:")
    for cls, p in sorted(all_proba.items(), key=lambda kv: -kv[1]):
        bar = "█" * int(p * 30)
        print(f"    {cls:<10} {p:6.2%}  {bar}")
    print("-" * 60)


def run_test_mode():
    """Прогон по заранее подготовленным профилям."""
    print("\n" + "=" * 70)
    print("ТЕСТИРОВАНИЕ НА ЗАДАННЫХ ПРОФИЛЯХ")
    print("=" * 70)

    correct = 0
    results = []

    for i, item in enumerate(TEST_PROFILES, 1):
        pred, prob, all_proba = predict_profile(item["profile"])
        ok = (pred == item["expected"])
        correct += int(ok)

        results.append({
            "№": i,
            "Профиль": item["name"],
            "Ожидалось": item["expected"],
            "Прогноз": pred,
            "Уверенность": f"{prob:.1%}",
            "Верно?": "✅" if ok else "❌",
        })

        print(f"\n[{i}/{len(TEST_PROFILES)}] {item['name']}")
        print(f"  Ожидалось: {item['expected']}")
        print(f"  Прогноз:   {pred}  (уверенность {prob:.1%})")
        print(f"  Результат: {'✅ ВЕРНО' if ok else '❌ ОШИБКА'}")

    total = len(TEST_PROFILES)
    print("\n" + "=" * 70)
    print(f"ИТОГ: {correct}/{total} правильных ({correct/total:.1%})")
    print("=" * 70)

    # Сводная таблица
    print("\nСВОДНАЯ ТАБЛИЦА:")
    print(pd.DataFrame(results).to_string(index=False))

    # Сохраняем отчёт
    report_path = os.path.join(OUT_DIR, "test_report.csv")
    pd.DataFrame(results).to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"\nОтчёт сохранён: {os.path.abspath(report_path)}")


# ------------------------------------------------------------------
# 13. ВАЖНОСТЬ ПРИЗНАКОВ (сохраняем до меню, чтобы было в отчёте)
# ------------------------------------------------------------------
perm = permutation_importance(
    best_model, X_test, y_test,
    n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
)
importances = pd.Series(perm.importances_mean, index=X_test.columns)
top = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 7))
sns.barplot(x=top.values, y=top.index, hue=top.index,
            palette="viridis", legend=False)
plt.title("Топ-15 важных признаков (permutation importance)")
plt.xlabel("Среднее падение accuracy")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150)
plt.close()

print("=" * 70)
print("ТОП-10 ВАЖНЫХ ПРИЗНАКОВ:")
for name, val in top.head(10).items():
    print(f"  {val:+.4f}  {name}")
print(f"\nГрафики сохранены в: {os.path.abspath(OUT_DIR)}")

# ==================================================================
# 14. ГЛАВНОЕ МЕНЮ
# ==================================================================
def main_menu():
    while True:
        print("\n" + "=" * 70)
        print("ГЛАВНОЕ МЕНЮ")
        print("=" * 70)
        print("  1 — Демонстрация (ввести профиль нового пользователя)")
        print("  2 — Тестирование на заданных профилях")
        print("  3 — Показать тестовые профили (без прогона)")
        print("  0 — Выход")
        choice = input("Ваш выбор: ").strip()

        if choice == "1":
            run_demo_mode()
        elif choice == "2":
            run_test_mode()
        elif choice == "3":
            print("\nДОСТУПНЫЕ ТЕСТОВЫЕ ПРОФИЛИ:")
            for i, item in enumerate(TEST_PROFILES, 1):
                print(f"  {i}. {item['name']} (ожидается: {item['expected']})")
        elif choice == "0":
            print("\nВыход. До свидания!")
            break
        else:
            print("⚠ Неверный ввод. Попробуйте снова.")


if __name__ == "__main__":
    main_menu()