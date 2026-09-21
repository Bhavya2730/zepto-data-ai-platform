# Data Pipeline

## Objective

Scrape book catalogue data from `books.toscrape.com`, clean and type the fields, convert GBP prices to INR using the required fixed project rate, load a normalized SQLite database, and validate SQL results with pandas.

## Files

- `scrape_pipeline.py` — scraping, cleaning, conversion, and SQLite loading.
- `queries.py` — required SQL queries plus `pd.read_sql` and `pd.merge` validation.
- `output/clean_books.csv` — generated cleaned dataset.
- `output/books.db` — generated SQLite database.

## Run

From the repository root:

```bash
python data_pipeline/scrape_pipeline.py
python data_pipeline/queries.py
```

The scraper automatically follows pagination for the first three categories it discovers and checks that the final dataset contains at least 60 books across at least three categories.

## Cleaning decisions

- `price_gbp`: removes `£` and converts to `float`.
- `rating`: maps `One`–`Five` to integers 1–5.
- `in_stock`: parses the availability text into a Boolean.
- Numeric parsing failures are median-imputed.
- Rows missing essential `title` or `category` values are dropped rather than inserted with unusable identifiers.
- `price_inr` uses the fixed assignment baseline **1 GBP = 105.50 INR**. No live exchange-rate lookup is used.

## Database design

`categories` contains one row per category and has `category_id` as its primary key.

`books` contains catalogue records and references `categories.category_id` through a foreign key. This separates category data from book records and avoids repeating category names in every database row.

## SQL coverage

The query script demonstrates:

- `SELECT` / `WHERE`
- `ORDER BY`
- `LIMIT`
- `DISTINCT`
- `BETWEEN`
- `JOIN`

It also reads multiple SQL results with `pd.read_sql()` and independently recreates the JOIN using `pd.merge()`, then checks that the results are equivalent.
