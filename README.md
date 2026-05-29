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


### نصب با mirror وقتی سرور به PyPI وصل نیست
میرور پیشنهادی برای pip:
```text
https://pypi.tuna.tsinghua.edu.cn/simple
```

نصب یک‌باره با mirror:
```bash
python -m pip install -r requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn
```

یا mirror را دائمی کن تا بعداً فقط `pip install -r requirements.txt` بزنی:
```bash
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
python -m pip install -r requirements.txt
```

نمونه تنظیمات آماده هم در `pip.conf.example` هست. اگر فقط API/crawler/chat را اجرا می‌کنی، فعلاً به NumPy نیاز نداری؛ ولی برای `train-text`, `eval-text`, `src.train`, `src.infer` باید `requirements.txt` نصب شود.

### 1) جمع‌آوری خودکار از اینترنت
```bash
python -m src.cli collect-text \
  --mode urls \
  --input data/seed_urls.txt \
  --out-csv data/raw_web_text.csv \
  --min-chars 100 \
  --verbose \
  --ignore-robots
```


### 1.1) اگر همه URLها با `[skip robots]` رد شدند
برای تست و جمع‌آوری اولیه می‌توانی robots را موقتاً نادیده بگیری:
```bash
python -m src.cli collect-text --mode urls --input data/seed_urls.txt --out-csv data/raw_web_text.csv --min-chars 100 --verbose --ignore-robots
```

### 1.2) یادگیری از Persian_instruct_dataset
1) ریپو را clone کن:
```bash
git clone https://github.com/mostafaamiri/Persian_instruct_dataset data/Persian_instruct_dataset
```
2) داده instruct را به حافظه چت تبدیل کن:
```bash
python -m src.cli import-instruct --repo-dir data/Persian_instruct_dataset --memory-path data/chat_memory.json
```
3) حالا چت از این حافظه پاسخ می‌دهد:
```bash
python -m src.cli chat --message "هوش مصنوعی را ساده توضیح بده" --memory-path data/chat_memory.json
```

### 1.3) خزیدن پیوسته (infinite-style) با کشف لینک‌های جدید
این حالت از seed شروع می‌کند، از هر صفحه لینک جدید پیدا می‌کند، و ادامه می‌دهد.
هر ۱۰۰ لینک (یا مقدار `--ask-every`) از شما می‌پرسد ادامه بدهد یا نه:
```bash
python -m src.cli crawl-learn \
  --input data/seed_urls.txt \
  --out-csv data/raw_web_text.csv \
  --min-chars 100 \
  --ask-every 100 \
  --workers 16 \
  --delay-sec 0 \
  --db-path data/netai.db \
  --save-every 500 \
  --verbose \
  --ignore-robots
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


### 6) شروع مکالمه و یادگیری از حرف‌های کاربر
- پاسخ گرفتن:
```bash
python -m src.cli chat --message "سلام، در مورد یادگیری ماشین توضیح بده"
```
- پاسخ خودکار از کورپوس جمع‌آوری‌شده (بدون learn-chat دستی):
```bash
python -m src.cli chat --message "یادگیری ماشین چیست؟" --corpus-csv data/raw_web_text.csv --top-k 3 --min-similarity 0.15
```

> نکته: برای پیام‌های خیلی کوتاه مثل «سلام»، بات دیگر حافظه/کورپوس نامرتبط را انتخاب نمی‌کند و پاسخ greeting داخلی می‌دهد. برای جلوگیری از جواب‌های پرت، پاسخ‌های memory/corpus باید از آستانه `--min-similarity` عبور کنند.
- یاد دادن یک پاسخ جدید به بات (حافظه‌محور):
```bash
python -m src.cli learn-chat   --user-message "چطور مدل رو بهتر کنم؟"   --assistant-message "داده بیشتر، ارزیابی درست، و تنظیم هایپرپارامترها را انجام بده."
```


### 6.1) اجرای API روی سرور
```bash
python -m src.cli serve-api \
  --host 0.0.0.0 \
  --port 8000 \
  --corpus-csv data/raw_web_text.csv \
  --db-path data/netai.db
```

نمونه پرسیدن سوال از API:
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"یادگیری ماشین چیست؟","top_k":3,"min_similarity":0.15}'
```

وضعیت پیشرفت جمع‌آوری داده:
```bash
curl http://127.0.0.1:8000/stats
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


### Install with a mirror when the server cannot reach PyPI
Recommended pip mirror:
```text
https://pypi.tuna.tsinghua.edu.cn/simple
```

One-time install with the mirror:
```bash
python -m pip install -r requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn
```

Or configure pip once, then keep using normal install commands:
```bash
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
python -m pip install -r requirements.txt
```

A ready-to-copy example is also included in `pip.conf.example`. If you only run API/crawler/chat, NumPy is not needed immediately; `requirements.txt` is required for `train-text`, `eval-text`, `src.train`, and `src.infer`.

### 1) Auto-collect from web
```bash
python -m src.cli collect-text --mode urls --input data/seed_urls.txt --out-csv data/raw_web_text.csv --min-chars 100 --verbose
# اگر robots.txt در دسترس نباشد، پیام [robots unavailable] می‌بینی ولی crawl ادامه می‌یابد.
```

### 1.1) Faster continuous crawling with link discovery
```bash
python -m src.cli crawl-learn --input data/seed_urls.txt --out-csv data/raw_web_text.csv --min-chars 100 --ask-every 100 --workers 16 --delay-sec 0 --db-path data/netai.db --save-every 500 --verbose --ignore-robots
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


### 6) Start chatting and incremental learning
- Get a reply:
```bash
python -m src.cli chat --message "Explain machine learning basics"
```
- Auto-answer from the crawled corpus (no manual learn-chat required):
```bash
python -m src.cli chat --message "What is machine learning?" --corpus-csv data/raw_web_text.csv --top-k 3 --min-similarity 0.15
```

> Note: For very short greetings such as `hi`, the bot now returns a built-in greeting instead of matching unrelated memory/corpus rows. Memory/corpus answers must pass `--min-similarity` to avoid irrelevant replies.
- Teach a new pair into memory:
```bash
python -m src.cli learn-chat   --user-message "How can I improve the model?"   --assistant-message "Use more data, evaluate correctly, and tune hyperparameters."
```


### 6.1) Run the API on a server
```bash
python -m src.cli serve-api --host 0.0.0.0 --port 8000 --corpus-csv data/raw_web_text.csv --db-path data/netai.db
```

Ask a question through the API:
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is machine learning?","top_k":3,"min_similarity":0.15}'
```

Check data-collection progress:
```bash
curl http://127.0.0.1:8000/stats
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
