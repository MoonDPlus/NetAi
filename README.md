# NetAi

شروع یک پروژه هوش مصنوعی مستقل با پایتون (بدون وابستگی به APIهای آماده).

## مرحله فعلی (گام دوم)
در این مرحله یک شبکه عصبی ساده (MLP) را از صفر با `NumPy` توسعه داده‌ایم:

- لایه Dense
- فعال‌ساز ReLU
- فعال‌ساز Sigmoid
- Binary Cross Entropy Loss
- گرادیان و Backpropagation
- آموزش **Mini-batch**
- Split آموزش/اعتبارسنجی
- ذخیره/لود وزن‌ها
- تست‌های پایه

## ساختار پروژه

```text
.
├── README.md
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── data.py
│   ├── mlp.py
│   └── train.py
└── tests
    ├── test_data.py
    └── test_model.py
```

## نصب و اجرا

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --epochs 1500 --lr 0.1 --batch-size 32 --repeats 300 --noise 0.05 --save-path model.json
```

## اجرای تست‌ها

```bash
python -m unittest discover -s tests
```

## هدف گام بعدی
- پیاده‌سازی optimizerهای بهتر (Momentum/Adam)
- اضافه کردن early stopping
- اضافه کردن متریک‌ها و گزارش دقیق‌تر
