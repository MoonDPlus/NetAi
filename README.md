# NetAi

پروژه هوش مصنوعی مستقل با پایتون (بدون API آماده).

## الان این پروژه چه کار می‌تواند بکند؟
- پیاده‌سازی MLP از صفر با NumPy (Dense/ReLU/Sigmoid/BCE + backprop)
- آموزش/ارزیابی روی مسئله XOR
- مسیر طبقه‌بندی متن فارسی با Bag-of-Words
- Train/Val/Test split واقعی برای متن
- CLI یکپارچه برای train/eval متن
- متریک‌های classification شامل accuracy/precision/recall/F1 + confusion matrix
- ذخیره history آموزش در JSON

- جمع‌آوری خودکار متن خام از وب (با رعایت robots.txt)

## نصب و اجرا (فارسی)
### 1) ساخت محیط و نصب وابستگی‌ها
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) آموزش مدل متن فارسی (با split واقعی)
```bash
python -m src.cli train-text \
  --data data/sample_sentiment_fa.csv \
  --model-path text_model.json \
  --vocab-path vocab.json \
  --history-path text_history.json \
  --optimizer adam \
  --val-ratio 0.2 \
  --test-ratio 0.2 \
  --patience 50
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

### 6) جمع‌آوری خودکار دیتاست خام از اینترنت
ابتدا یک فایل متنی بساز که در هر خط یک URL داشته باشد (مثلا `urls.txt`) سپس:
```bash
python -m src.cli collect-text \
  --mode urls \
  --input urls.txt \
  --out-csv data/raw_web_text.csv
```

یا اگر لیست sitemap داری (`sitemaps.txt`):
```bash
python -m src.cli collect-text \
  --mode sitemaps \
  --input sitemaps.txt \
  --out-csv data/raw_web_text.csv
```

---

## Installation & Usage (English)
### 1) Create virtual environment and install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Train the Persian text classifier (with real split)
```bash
python -m src.cli train-text \
  --data data/sample_sentiment_fa.csv \
  --model-path text_model.json \
  --vocab-path vocab.json \
  --history-path text_history.json \
  --optimizer adam \
  --val-ratio 0.2 \
  --test-ratio 0.2 \
  --patience 50
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

### 6) Auto-collect raw text dataset from the web
Create a text file with one URL per line (for example `urls.txt`) and run:
```bash
python -m src.cli collect-text \
  --mode urls \
  --input urls.txt \
  --out-csv data/raw_web_text.csv
```

Or with sitemap list (`sitemaps.txt`):
```bash
python -m src.cli collect-text \
  --mode sitemaps \
  --input sitemaps.txt \
  --out-csv data/raw_web_text.csv
```


### 7) یادگیری خودکار از اینترنت + گزارش ظرفیت واژگان/جمله
1. از لیست URL اولیه (`data/seed_urls.txt`) متن جمع کن:
```bash
python -m src.cli collect-text --mode urls --input data/seed_urls.txt --out-csv data/raw_web_text.csv --min-chars 100 --verbose
```
2. آمار یادگیری را بگیر (کل کلمات، کلمات یکتا، تعداد جمله‌ها، ظرفیت تقریبی تولید):
```bash
python -m src.cli analyze-corpus --data-csv data/raw_web_text.csv --out-json data/corpus_stats.json
```

## Testing
```bash
python -m unittest discover -s tests
```
