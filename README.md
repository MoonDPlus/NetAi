# NetAi

پروژه هوش مصنوعی مستقل با پایتون (بدون API آماده).

## مرحله فعلی (گام چهارم)
علاوه بر مسیر XOR، حالا یک مسیر واقعی‌تر برای طبقه‌بندی متن هم داریم:

- MLP از صفر با NumPy
- train/infer روی XOR
- CLI یکپارچه برای train/eval متن
- Bag-of-Words ساده برای متن فارسی
- متریک‌های accuracy/precision/recall/F1

## داده نمونه
`data/sample_sentiment_fa.csv` شامل نمونه‌های کوچک sentiment فارسی است.

## آموزش مدل متن
```bash
python -m src.cli train-text \
  --data data/sample_sentiment_fa.csv \
  --model-path text_model.json \
  --vocab-path vocab.json \
  --optimizer adam
```

## ارزیابی مدل متن
```bash
python -m src.cli eval-text \
  --data data/sample_sentiment_fa.csv \
  --model-path text_model.json \
  --vocab-path vocab.json
```

## مرحله بعدی
- جایگزینی BoW با embedding بهتر
- اضافه‌کردن mini-batch و early stopping به train-text
- افزایش داده واقعی
