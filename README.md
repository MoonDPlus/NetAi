# NetAi

پروژه هوش مصنوعی مستقل با پایتون (بدون API آماده).

## مرحله پیشرفته‌تر (Phase Advanced)
در این مرحله، علاوه بر آموزش مدل، یک چرخه کامل **جمع‌آوری → تحلیل → تولید جمله** اضافه شده است:
- Crawl خودکار از اینترنت با رعایت `robots.txt` (در صورت خطای دسترسی به robots، حالت پیش‌فرض fail-open است تا کل جمع‌آوری قفل نشود)
- تحلیل آماری کورپوس (کلمات/جمله‌ها/ظرفیت تقریبی)
- تولید جمله‌های جدید بر پایه bigram chain از کورپوس جمع‌آوری‌شده


### دیتاست‌های پیشنهادی برای شروع (انتخاب‌شده)
- لیست curated در `data/datasets_catalog.md` اضافه شد (Sentiment + Corpus عمومی فارسی).
- برای crawl اولیه، `data/seed_urls.txt` بزرگ‌تر شده تا پوشش موضوعی بهتر شود.

### جریان پیشنهادی عملی
1) یک دیتاست برچسب‌دار از catalog انتخاب کن و به قالب `text,label` تبدیل کن.
2) در کنار آن با `collect-text` یک کورپوس بزرگ unlabeled بساز.
3) با `analyze-corpus` وضعیت یادگیری (کلمه/جمله/ظرفیت) را بعد از هر راند ثبت کن.
4) با `generate-text` کیفیت تقریبی یادگیری زبانی را بررسی کن.

## نصب و اجرا (فارسی)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) جمع‌آوری خودکار از اینترنت
```bash
python -m src.cli collect-text \
  --mode urls \
  --input data/seed_urls.txt \
  --out-csv data/raw_web_text.csv \
  --min-chars 100 \
  --verbose
```

### 2) تحلیل کورپوس یادگیری‌شده
```bash
python -m src.cli analyze-corpus \
  --data-csv data/raw_web_text.csv \
  --out-json data/corpus_stats.json
```
خروجی شامل این شاخص‌هاست:
- total_words
- unique_words
- total_sentences
- avg_sentence_len
- generation_estimate (گزینه‌های شروع جمله + گذار واژه)

### 3) تولید جمله از داده یادگرفته‌شده
```bash
python -m src.cli generate-text \
  --data-csv data/raw_web_text.csv \
  --n-sentences 10 \
  --max-words 14
```

### 4) آموزش/ارزیابی طبقه‌بندی متن
```bash
python -m src.cli train-text --data data/sample_sentiment_fa.csv --model-path text_model.json --vocab-path vocab.json --history-path text_history.json --optimizer adam --val-ratio 0.2 --test-ratio 0.2 --patience 50
python -m src.cli eval-text --data data/sample_sentiment_fa.csv --model-path text_model.json --vocab-path vocab.json
```

### 5) مسیر XOR
```bash
python -m src.train --epochs 1500 --lr 0.01 --batch-size 32 --repeats 300 --noise 0.05 --optimizer adam --patience 200 --save-path model.json
python -m src.infer --model-path model.json --x1 0 --x2 1
```

---


### Recommended starter datasets (curated)
- A curated list is included at `data/datasets_catalog.md` (Persian sentiment + general corpora).
- `data/seed_urls.txt` has been expanded for broader web-crawl bootstrapping.

### Practical workflow
1) Pick one labeled dataset from the catalog and normalize to `text,label`.
2) Build a large unlabeled corpus with `collect-text`.
3) Track learning progress with `analyze-corpus` after each crawl round.
4) Probe language learning quality with `generate-text`.

## Installation & Usage (English)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Auto-collect from web
```bash
python -m src.cli collect-text --mode urls --input data/seed_urls.txt --out-csv data/raw_web_text.csv --min-chars 100 --verbose
# اگر robots.txt در دسترس نباشد، پیام [robots unavailable] می‌بینی ولی crawl ادامه می‌یابد.
```

### 2) Analyze learned corpus
```bash
python -m src.cli analyze-corpus --data-csv data/raw_web_text.csv --out-json data/corpus_stats.json
```

### 3) Generate sentences from learned corpus
```bash
python -m src.cli generate-text --data-csv data/raw_web_text.csv --n-sentences 10 --max-words 14
```

### 4) Train/Eval text classifier
```bash
python -m src.cli train-text --data data/sample_sentiment_fa.csv --model-path text_model.json --vocab-path vocab.json --history-path text_history.json --optimizer adam --val-ratio 0.2 --test-ratio 0.2 --patience 50
python -m src.cli eval-text --data data/sample_sentiment_fa.csv --model-path text_model.json --vocab-path vocab.json
```

### 5) XOR path
```bash
python -m src.train --epochs 1500 --lr 0.01 --batch-size 32 --repeats 300 --noise 0.05 --optimizer adam --patience 200 --save-path model.json
python -m src.infer --model-path model.json --x1 0 --x2 1
```

## Testing
```bash
python -m unittest discover -s tests
```
