from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50
OUTPUT_DIR = Path(__file__).parent / "output"
DB_PATH = OUTPUT_DIR / "books.db"
CSV_PATH = OUTPUT_DIR / "clean_books.csv"

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


@dataclass
class Book:
    title: str
    price_gbp: float
    rating: int
    in_stock: bool
    category: str
    price_inr: float


def get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    response = session.get(url, timeout=20)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def get_all_products_page_urls(session: requests.Session, max_pages: int = 5) -> list[str]:
    """Return URLs for the first max_pages All Products pages."""
    urls = []
    page_url = requests.compat.urljoin(BASE_URL, "catalogue/page-1.html")

    for _ in range(max_pages):
        urls.append(page_url)
        soup = get_soup(session, page_url)
        next_link = soup.select_one("li.next a")
        if not next_link:
            break
        page_url = requests.compat.urljoin(page_url, next_link.get("href"))

    return urls


def parse_book(article, category: str) -> Optional[dict]:
    title_tag = article.select_one("h3 a")
    price_tag = article.select_one(".price_color")
    rating_tag = article.select_one("p.star-rating")
    availability_tag = article.select_one(".availability")

    if not all([title_tag, price_tag, rating_tag, availability_tag]):
        return None

    title = title_tag.get("title") or title_tag.get_text(strip=True)
    price_text = price_tag.get_text(strip=True)
    rating_classes = rating_tag.get("class", [])
    rating_text = next(
        (value for value in rating_classes if value in RATING_MAP),
        None,
    )
    availability_text = availability_tag.get_text(" ", strip=True)

    return {
        "title": title,
        "price_text": price_text,
        "rating_text": rating_text,
        "availability_text": availability_text,
        "category": category,
    }


def scrape_category(session: requests.Session, category: str, url: str) -> list[dict]:
    rows = []
    page_url = url

    while page_url:
        soup = get_soup(session, page_url)

        for article in soup.select("article.product_pod"):
            row = parse_book(article, category)
            if row:
                rows.append(row)

        next_link = soup.select_one("li.next a")
        page_url = (
            requests.compat.urljoin(page_url, next_link.get("href"))
            if next_link
            else None
        )

    return rows


def clean_data(raw_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_rows)

    # Parse GBP prices robustly from the scraped text.
    # The site displays prices such as "£51.77"; extracting the numeric
    # portion avoids locale/encoding issues with the pound symbol.
    df["price_gbp"] = pd.to_numeric(
        df["price_text"]
        .astype(str)
        .str.extract(r"([0-9]+(?:\\.[0-9]+)?)", expand=False),
        errors="coerce",
    )
    df["rating"] = df["rating_text"].map(RATING_MAP)
    df["in_stock"] = df["availability_text"].str.contains(
        "in stock", case=False, na=False
    )

    # Numeric parsing failures are median-imputed as required.
    if df["price_gbp"].isna().any():
        df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())

    if df["rating"].isna().any():
        df["rating"] = df["rating"].fillna(df["rating"].median()).round().astype(int)

    # Rows missing essential text/category information are dropped.
    df = df.dropna(subset=["title", "category"]).copy()

    df["price_inr"] = df["price_gbp"] * GBP_TO_INR

    result = df[
        ["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]
    ].copy()

    result["price_gbp"] = result["price_gbp"].astype(float)
    result["price_inr"] = result["price_inr"].astype(float)
    result["rating"] = result["rating"].astype(int)
    result["in_stock"] = result["in_stock"].astype(bool)

    return result.drop_duplicates(subset=["title", "category"]).reset_index(drop=True)


def load_sqlite(df: pd.DataFrame, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        conn.executescript(
            """
            DROP TABLE IF EXISTS books;
            DROP TABLE IF EXISTS categories;

            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY,
                category_name TEXT UNIQUE NOT NULL
            );

            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
                category_id INTEGER NOT NULL,
                FOREIGN KEY (category_id)
                    REFERENCES categories(category_id)
            );
            """
        )

        categories = sorted(df["category"].unique())
        conn.executemany(
            "INSERT INTO categories(category_name) VALUES (?)",
            [(category,) for category in categories],
        )

        category_ids = dict(
            conn.execute("SELECT category_name, category_id FROM categories").fetchall()
        )

        records = [
            (
                row.title,
                row.price_gbp,
                row.price_inr,
                row.rating,
                int(row.in_stock),
                category_ids[row.category],
            )
            for row in df.itertuples(index=False)
        ]

        conn.executemany(
            """
            INSERT INTO books
            (title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            records,
        )

        conn.commit()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; ZeptoCapstone/1.0)"}
        )

        # Crawl the first 5 paginated All Products pages.
        # This satisfies the assignment alternative and gives well over 60 books.
        page_urls = get_all_products_page_urls(session, max_pages=5)
        raw_rows: list[dict] = []

        for page_url in page_urls:
            soup = get_soup(session, page_url)
            for article in soup.select("article.product_pod"):
                # All Products pages do not expose category directly on the card.
                # Follow each book link and read its breadcrumb category.
                book_link = article.select_one("h3 a")
                if not book_link:
                    continue

                book_url = requests.compat.urljoin(page_url, book_link.get("href"))
                book_soup = get_soup(session, book_url)

                breadcrumb = book_soup.select("ul.breadcrumb li a")
                category = (
                    breadcrumb[-1].get_text(strip=True)
                    if breadcrumb
                    else "Unknown"
                )

                row = parse_book(article, category)
                if row:
                    raw_rows.append(row)

    df = clean_data(raw_rows)

    if df.empty:
        raise RuntimeError(
            "No books were parsed. Please check the website response and selectors."
        )

    if len(df) < 60:
        raise RuntimeError(
            f"Acceptance criteria not met: only {len(df)} books were scraped."
        )

    df.to_csv(CSV_PATH, index=False)
    load_sqlite(df, DB_PATH)

    print(f"Scraped rows: {len(raw_rows)}")
    print(f"Clean rows: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Saved CSV: {CSV_PATH}")
    print(f"Saved SQLite DB: {DB_PATH}")
    print(f"GBP → INR fixed rate: {GBP_TO_INR}")


if __name__ == "__main__":
    main()
