# NetAi

شروع یک پروژه هوش مصنوعی مستقل با پایتون (بدون وابستگی به APIهای آماده).

## مرحله فعلی (گام سوم)
در این مرحله نسخه قبلی را کاربردی‌تر کردیم:

- لایه Dense
- فعال‌ساز ReLU
- فعال‌ساز Sigmoid
- Binary Cross Entropy Loss
- گرادیان و Backpropagation
- آموزش **Mini-batch**
- Split آموزش/اعتبارسنجی
- ذخیره/لود وزن‌ها
- **Optimizer قابل انتخاب (SGD/Adam)**
- **Early Stopping**
- **اسکریپت inference**
- تست‌های پایه + تست Train

## ساختار پروژه

```text
.
├── README.md
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── data.py
│   ├── infer.py
│   ├── mlp.py
│   └── train.py
└── tests
    ├── test_data.py
    ├── test_model.py
    └── test_train.py
```

## نصب و اجرا

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --epochs 1500 --lr 0.01 --batch-size 32 --repeats 300 --noise 0.05 --optimizer adam --patience 200 --save-path model.json
```

## اجرای inference

```bash
python -m src.infer --model-path model.json --x1 0 --x2 1
```

## اجرای تست‌ها

```bash
python -m unittest discover -s tests
```

## هدف گام بعدی
- اضافه کردن چند وظیفه واقعی‌تر از XOR (مثلاً طبقه‌بندی ساده متن)
- ساخت یک CLI یکپارچه برای train/eval/infer
- گزارش‌گیری بهتر (ذخیره history و نمودار)
