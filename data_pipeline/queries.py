from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


DB_PATH = Path(__file__).parent / "output" / "books.db"


QUERIES = {
    "select_where": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating >= 4
        LIMIT 10;
    """,
    "order_by_limit": """
        SELECT title, price_gbp, rating
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
    """,
    "distinct": """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name;
    """,
    "between": """
        SELECT title, price_gbp, price_inr
        FROM books
        WHERE price_gbp BETWEEN 20 AND 30
        ORDER BY price_gbp;
    """,
    "join": """
        SELECT
            c.category_name,
            b.title,
            b.rating,
            b.price_gbp,
            b.in_stock
        FROM books AS b
        JOIN categories AS c
            ON b.category_id = c.category_id
        ORDER BY b.rating DESC, b.price_gbp DESC, b.title ASC
        LIMIT 15;
    """,
}


def run_queries() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        print("=" * 80)
        print("SQL QUERY OUTPUTS")
        print("=" * 80)

        for name, query in QUERIES.items():
            print(f"\n--- {name} ---")
            print(query.strip())
            result = pd.read_sql(query, conn)
            print(result.to_string(index=False))

        # Read at least two query results through pd.read_sql.
        result_1 = pd.read_sql(QUERIES["select_where"], conn)
        result_2 = pd.read_sql(QUERIES["order_by_limit"], conn)
        sql_join = pd.read_sql(QUERIES["join"], conn)

        # Separately reproduce the JOIN using pd.merge on in-memory DataFrames.
        books_df = pd.read_sql(
            """
            SELECT book_id, title, rating, price_gbp, in_stock, category_id
            FROM books
            """,
            conn,
        )
        categories_df = pd.read_sql(
            "SELECT category_id, category_name FROM categories",
            conn,
        )

        merge_join = (
            pd.merge(books_df, categories_df, on="category_id", how="inner")
            [["category_name", "title", "rating", "price_gbp", "in_stock"]]
            .sort_values(["rating", "price_gbp", "title"],ascending=[False, False, True])
            .head(15)
            .reset_index(drop=True)
        )

        # SQLite stores booleans as INTEGER (0/1), while the cleaned
        # DataFrame uses Python booleans. Normalize both sides before
        # comparing so we compare the actual JOIN values, not pandas dtypes.
        sql_join_compare = sql_join.reset_index(drop=True).copy()
        sql_join_compare["in_stock"] = sql_join_compare["in_stock"].astype(bool)

        merge_join["in_stock"] = merge_join["in_stock"].astype(bool)

        # Normalize numeric dtypes as well.
        sql_join_compare["rating"] = sql_join_compare["rating"].astype(int)
        merge_join["rating"] = merge_join["rating"].astype(int)
        sql_join_compare["price_gbp"] = sql_join_compare["price_gbp"].astype(float)
        merge_join["price_gbp"] = merge_join["price_gbp"].astype(float)

        print("\n--- pd.read_sql JOIN vs pd.merge JOIN ---")
        comparison = pd.concat(
            [
                sql_join_compare.add_prefix("sql_"),
                merge_join.add_prefix("merge_"),
            ],
            axis=1,
        )
        print(comparison.to_string(index=False))

        # Compare the actual JOIN values after normalizing dtypes and indexes.
        sql_values = sql_join_compare.astype({
            "category_name": "string",
            "title": "string",
            "rating": "int64",
            "price_gbp": "float64",
            "in_stock": "bool",
        }).reset_index(drop=True)

        merge_values = merge_join.astype({
            "category_name": "string",
            "title": "string",
            "rating": "int64",
            "price_gbp": "float64",
            "in_stock": "bool",
        }).reset_index(drop=True)

        equivalent = sql_values.to_numpy().tolist() == merge_values.to_numpy().tolist()

        print(f"\nJOIN outputs equivalent: {equivalent}")

        if not equivalent:
            raise AssertionError("SQL JOIN and pandas merge results do not match.")


if __name__ == "__main__":
    run_queries()
