# NetAi

پروژه هوش مصنوعی مستقل با پایتون (بدون API آماده).

## الان این پروژه چه کار می‌تواند بکند؟
- پیاده‌سازی MLP از صفر با NumPy (Dense/ReLU/Sigmoid/BCE + backprop)
- آموزش/ارزیابی روی مسئله XOR
- مسیر طبقه‌بندی متن فارسی با Bag-of-Words
- CLI یکپارچه برای train/eval متن
- متریک‌های classification شامل accuracy/precision/recall/F1

## نصب و اجرا (فارسی)
### 1) ساخت محیط و نصب وابستگی‌ها
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) آموزش مدل متن فارسی
```bash
python -m src.cli train-text \
  --data data/sample_sentiment_fa.csv \
  --model-path text_model.json \
  --vocab-path vocab.json \
  --optimizer adam
```

### 3) ارزیابی مدل متن
```bash
python -m src.cli eval-text \
  --data data/sample_sentiment_fa.csv \
  --model-path text_model.json \
  --vocab-path vocab.json
```

### 4) آموزش مسیر XOR
```bash
python -m src.train --epochs 1500 --lr 0.01 --batch-size 32 --repeats 300 --noise 0.05 --optimizer adam --patience 200 --save-path model.json
```

### 5) اجرای inference برای XOR
```bash
python -m src.infer --model-path model.json --x1 0 --x2 1
```

---

## Installation & Usage (English)
### 1) Create virtual environment and install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Train the Persian text classifier
```bash
python -m src.cli train-text \
  --data data/sample_sentiment_fa.csv \
  --model-path text_model.json \
  --vocab-path vocab.json \
  --optimizer adam
```

### 3) Evaluate the text classifier
```bash
python -m src.cli eval-text \
  --data data/sample_sentiment_fa.csv \
  --model-path text_model.json \
  --vocab-path vocab.json
```

### 4) Train the XOR pipeline
```bash
python -m src.train --epochs 1500 --lr 0.01 --batch-size 32 --repeats 300 --noise 0.05 --optimizer adam --patience 200 --save-path model.json
```

### 5) Run XOR inference
```bash
python -m src.infer --model-path model.json --x1 0 --x2 1
```

## Testing
```bash
python -m unittest discover -s tests
```
