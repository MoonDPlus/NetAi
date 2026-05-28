# Persian NLP Dataset Catalog (Curated)

> Added on May 28, 2026.

## Sentiment Analysis

1. **SentiPers / Digikala-based sentiment resources (Hooshvare)**  
   Docs: https://hooshvare.github.io/docs/datasets/sa

2. **ParsiAI Digikala Sentiment (Hugging Face Dataset)**  
   Link: https://huggingface.co/datasets/ParsiAI/digikala-sentiment-analysis

3. **Persian Sentiment Dataset Index (nlpdataset.ir)**  
   Link: https://nlpdataset.ir/farsi/sentiment_analysis.html

## General Persian Text Corpora

4. **MirasText (ACL Anthology)**  
   Paper: https://aclanthology.org/L18-1188/

5. **Hamshahri Corpus (overview)**  
   Link: https://en.wikipedia.org/wiki/Hamshahri_Corpus

## Notes for NetAi

- Prefer datasets with clear licensing and citations.
- Keep raw downloaded files out of git when large; store only scripts/metadata.
- For training in this repo, convert all selected datasets to a normalized CSV schema:
  - classification: `text,label`
  - unlabeled corpus: `url,text`
