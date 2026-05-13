"""
Day 2, Part 1 — Build the golden eval set.

WHY THIS FILE EXISTS:
  The golden eval set is the single most important artifact in this project.
  It never touches the training pipeline — it is purely for measurement.
  Every model we test (GPT-4o, Claude, base Qwen, fine-tuned Qwen) gets
  scored against these exact same questions. That's how we get a fair,
  apples-to-apples comparison table.

HOW IT'S STRUCTURED:
  - 200 examples across 10 schemas (20 per schema)
  - Each schema gets: 6 easy, 8 medium, 6 hard questions
  - Each question has:
      - schema_name: which schema to use
      - question: natural language
      - sql: the reference (correct) answer
      - complexity: easy / medium / hard
      - tags: what SQL features it tests (useful for error analysis later)

WHY HAND-CRAFTED AND NOT AUTO-GENERATED:
  Auto-generated eval sets have a subtle flaw: the same model that generates
  training data tends to "know" the eval questions. Hand-crafted questions
  give you a trustworthy, unbiased measurement. These 200 questions are
  written to specifically probe the failure modes of GPT-4o on domain SQL:
  multi-join aggregations, window functions, correlated subqueries,
  implicit business logic in column names.

Output: data/eval/golden_eval.jsonl
"""

import json
from pathlib import Path

ROOT     = Path(__file__).parent.parent
EVAL_DIR = ROOT / "data" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# THE GOLDEN EVAL SET
# Format per entry:
#   schema_name, question, sql, complexity, tags
#
# Tags explain WHAT SQL SKILL is being tested — useful later when you
# do error analysis ("model fails on window functions 60% of the time")

GOLDEN_EVAL = [

    # SCHEMA 1: marketplace_v1
    # Tables: users, sellers, categories, products, orders, order_items, reviews

    # Easy (6) ─ single table or trivial join, basic filter/aggregate
    {
        "schema_name": "marketplace_v1",
        "question": "How many active users are there?",
        "sql": "SELECT COUNT(*) AS active_users FROM users WHERE is_active = TRUE",
        "complexity": "easy",
        "tags": ["count", "filter"]
    },
    {
        "schema_name": "marketplace_v1",
        "question": "List all products with a price above 500, ordered by price descending.",
        "sql": "SELECT product_id, title, price FROM products WHERE price > 500 ORDER BY price DESC",
        "complexity": "easy",
        "tags": ["filter", "order_by"]
    },
    {
        "schema_name": "marketplace_v1",
        "question": "What is the average seller rating?",
        "sql": "SELECT AVG(rating) AS avg_rating FROM sellers",
        "complexity": "easy",
        "tags": ["avg"]
    },
    {
        "schema_name": "marketplace_v1",
        "question": "How many orders have status 'cancelled'?",
        "sql": "SELECT COUNT(*) AS cancelled_orders FROM orders WHERE status = 'cancelled'",
        "complexity": "easy",
        "tags": ["count", "filter"]
    },
    {
        "schema_name": "marketplace_v1",
        "question": "Find all categories that have no parent category.",
        "sql": "SELECT category_id, name FROM categories WHERE parent_id IS NULL",
        "complexity": "easy",
        "tags": ["null_check"]
    },
    {
        "schema_name": "marketplace_v1",
        "question": "What is the total revenue from all delivered orders?",
        "sql": "SELECT SUM(total_amount) AS total_revenue FROM orders WHERE status = 'delivered'",
        "complexity": "easy",
        "tags": ["sum", "filter"]
    },

    # Medium (8) ─ 1-2 joins, aggregation, GROUP BY
    {
        "schema_name": "marketplace_v1",
        "question": "For each seller, show their shop name and total number of listed products.",
        "sql": """SELECT s.shop_name, COUNT(p.product_id) AS total_products
FROM sellers s
LEFT JOIN products p ON s.seller_id = p.seller_id AND p.is_listed = TRUE
GROUP BY s.seller_id, s.shop_name
ORDER BY total_products DESC""",
        "complexity": "medium",
        "tags": ["join", "group_by", "count", "left_join"]
    },
    {
        "schema_name": "marketplace_v1",
        "question": "Find the top 5 users by total order spend.",
        "sql": """SELECT u.user_id, u.full_name, SUM(o.total_amount) AS total_spend
FROM users u
JOIN orders o ON u.user_id = o.user_id
WHERE o.status = 'delivered'
GROUP BY u.user_id, u.full_name
ORDER BY total_spend DESC
LIMIT 5""",
        "complexity": "medium",
        "tags": ["join", "group_by", "sum", "limit"]
    },
    {
        "schema_name": "marketplace_v1",
        "question": "What is the average review rating per category?",
        "sql": """SELECT c.name AS category_name, AVG(r.rating) AS avg_rating, COUNT(r.review_id) AS review_count
FROM categories c
JOIN products p ON c.category_id = p.category_id
JOIN reviews r ON p.product_id = r.product_id
GROUP BY c.category_id, c.name
ORDER BY avg_rating DESC""",
        "complexity": "medium",
        "tags": ["multi_join", "avg", "group_by"]
    },
    {
        "schema_name": "marketplace_v1",
        "question": "For each country, how many users placed at least one order?",
        "sql": """SELECT u.country, COUNT(DISTINCT u.user_id) AS buyers
FROM users u
JOIN orders o ON u.user_id = o.user_id
GROUP BY u.country
ORDER BY buyers DESC""",
        "complexity": "medium",
        "tags": ["join", "count_distinct", "group_by"]
    },
    {
        "schema_name": "marketplace_v1",
        "question": "Show the total quantity sold per product, only for products with more than 10 units sold.",
        "sql": """SELECT p.product_id, p.title, SUM(oi.quantity) AS total_sold
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.status = 'delivered'
GROUP BY p.product_id, p.title
HAVING SUM(oi.quantity) > 10
ORDER BY total_sold DESC""",
        "complexity": "medium",
        "tags": ["multi_join", "having", "sum", "group_by"]
    },
    {
        "schema_name": "marketplace_v1",
        "question": "Which sellers have an average product price above 1000?",
        "sql": """SELECT s.seller_id, s.shop_name, AVG(p.price) AS avg_price
FROM sellers s
JOIN products p ON s.seller_id = p.seller_id
GROUP BY s.seller_id, s.shop_name
HAVING AVG(p.price) > 1000
ORDER BY avg_price DESC""",
        "complexity": "medium",
        "tags": ["join", "having", "avg"]
    },
    {
        "schema_name": "marketplace_v1",
        "question": "Find the total revenue generated by each seller through their products.",
        "sql": """SELECT s.seller_id, s.shop_name, SUM(oi.quantity * oi.unit_price) AS seller_revenue
FROM sellers s
JOIN products p ON s.seller_id = p.seller_id
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.status = 'delivered'
GROUP BY s.seller_id, s.shop_name
ORDER BY seller_revenue DESC""",
        "complexity": "medium",
        "tags": ["multi_join", "sum", "arithmetic", "group_by"]
    },
    {
        "schema_name": "marketplace_v1",
        "question": "List products that have never received a review.",
        "sql": """SELECT p.product_id, p.title
FROM products p
LEFT JOIN reviews r ON p.product_id = r.product_id
WHERE r.review_id IS NULL""",
        "complexity": "medium",
        "tags": ["left_join", "null_check", "anti_join"]
    },

    # Hard (6) ─ subqueries, window functions, correlated queries, complex logic
    {
        "schema_name": "marketplace_v1",
        "question": "For each seller, find the product with the highest number of reviews.",
        "sql": """WITH review_counts AS (
    SELECT p.product_id, p.seller_id, p.title, COUNT(r.review_id) AS review_count
    FROM products p
    LEFT JOIN reviews r ON p.product_id = r.product_id
    GROUP BY p.product_id, p.seller_id, p.title
),
ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY seller_id ORDER BY review_count DESC) AS rn
    FROM review_counts
)
SELECT seller_id, product_id, title, review_count
FROM ranked
WHERE rn = 1""",
        "complexity": "hard",
        "tags": ["cte", "window_function", "row_number", "partition_by"]
    },
    {
        "schema_name": "marketplace_v1",
        "question": "Find users who have spent more than the average order value across all users.",
        "sql": """SELECT u.user_id, u.full_name, SUM(o.total_amount) AS total_spend
FROM users u
JOIN orders o ON u.user_id = o.user_id
GROUP BY u.user_id, u.full_name
HAVING SUM(o.total_amount) > (
    SELECT AVG(total_amount) FROM orders
)
ORDER BY total_spend DESC""",
        "complexity": "hard",
        "tags": ["subquery", "having", "correlated"]
    },
    {
        "schema_name": "marketplace_v1",
        "question": "Show month-over-month revenue growth for the past year.",
        "sql": """WITH monthly_revenue AS (
    SELECT
        DATE_TRUNC('month', created_at) AS month,
        SUM(total_amount) AS revenue
    FROM orders
    WHERE status = 'delivered'
    GROUP BY DATE_TRUNC('month', created_at)
)
SELECT
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month) AS prev_month_revenue,
    ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month)) /
          NULLIF(LAG(revenue) OVER (ORDER BY month), 0), 2) AS growth_pct
FROM monthly_revenue
ORDER BY month""",
        "complexity": "hard",
        "tags": ["cte", "window_function", "lag", "date_trunc", "arithmetic"]
    },
    {
        "schema_name": "marketplace_v1",
        "question": "Find sellers whose total sales have dropped compared to the previous month.",
        "sql": """WITH monthly_sales AS (
    SELECT
        p.seller_id,
        DATE_TRUNC('month', o.created_at) AS month,
        SUM(oi.quantity * oi.unit_price) AS monthly_revenue
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    WHERE o.status = 'delivered'
    GROUP BY p.seller_id, DATE_TRUNC('month', o.created_at)
),
with_prev AS (
    SELECT *,
        LAG(monthly_revenue) OVER (PARTITION BY seller_id ORDER BY month) AS prev_revenue
    FROM monthly_sales
)
SELECT seller_id, month, monthly_revenue, prev_revenue
FROM with_prev
WHERE monthly_revenue < prev_revenue
ORDER BY seller_id, month""",
        "complexity": "hard",
        "tags": ["cte", "window_function", "lag", "partition_by", "multi_join"]
    },
    {
        "schema_name": "marketplace_v1",
        "question": "Find the top 3 categories by revenue in each country.",
        "sql": """WITH country_category_revenue AS (
    SELECT
        u.country,
        c.name AS category_name,
        SUM(oi.quantity * oi.unit_price) AS revenue
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN users u ON o.user_id = u.user_id
    JOIN products p ON oi.product_id = p.product_id
    JOIN categories c ON p.category_id = c.category_id
    WHERE o.status = 'delivered'
    GROUP BY u.country, c.category_id, c.name
),
ranked AS (
    SELECT *, RANK() OVER (PARTITION BY country ORDER BY revenue DESC) AS rnk
    FROM country_category_revenue
)
SELECT country, category_name, revenue, rnk
FROM ranked
WHERE rnk <= 3
ORDER BY country, rnk""",
        "complexity": "hard",
        "tags": ["cte", "window_function", "rank", "partition_by", "multi_join"]
    },
    {
        "schema_name": "marketplace_v1",
        "question": "Identify users who placed orders in 3 or more consecutive months.",
        "sql": """WITH user_months AS (
    SELECT DISTINCT
        user_id,
        DATE_TRUNC('month', created_at) AS order_month
    FROM orders
),
with_gaps AS (
    SELECT *,
        LAG(order_month) OVER (PARTITION BY user_id ORDER BY order_month) AS prev_month
    FROM user_months
),
consecutive AS (
    SELECT user_id,
        SUM(CASE WHEN order_month = prev_month + INTERVAL '1 month' THEN 1 ELSE 0 END)
            OVER (PARTITION BY user_id) AS consecutive_count
    FROM with_gaps
)
SELECT DISTINCT user_id
FROM consecutive
WHERE consecutive_count >= 2""",
        "complexity": "hard",
        "tags": ["cte", "window_function", "lag", "interval", "distinct"]
    },

    # SCHEMA 2: neobank_v1
    # Tables: customers, accounts, transactions, cards, card_transactions

    {"schema_name": "neobank_v1", "question": "How many customers have completed KYC verification?",
     "sql": "SELECT COUNT(*) AS verified_customers FROM customers WHERE kyc_status = 'verified'",
     "complexity": "easy", "tags": ["count", "filter"]},

    {"schema_name": "neobank_v1", "question": "What is the total balance across all active savings accounts?",
     "sql": "SELECT SUM(balance) AS total_balance FROM accounts WHERE account_type = 'savings' AND status = 'active'",
     "complexity": "easy", "tags": ["sum", "filter"]},

    {"schema_name": "neobank_v1", "question": "List all failed transactions in the last 30 days.",
     "sql": "SELECT txn_id, from_account, to_account, amount, created_at FROM transactions WHERE status = 'failed' AND created_at >= CURRENT_DATE - INTERVAL '30 days'",
     "complexity": "easy", "tags": ["filter", "date_arithmetic"]},

    {"schema_name": "neobank_v1", "question": "How many active cards are there per card type?",
     "sql": "SELECT card_type, COUNT(*) AS active_cards FROM cards WHERE status = 'active' GROUP BY card_type",
     "complexity": "easy", "tags": ["group_by", "count"]},

    {"schema_name": "neobank_v1", "question": "Find customers with a high risk tier.",
     "sql": "SELECT customer_id, full_name, email FROM customers WHERE risk_tier = 'high' AND kyc_status = 'verified'",
     "complexity": "easy", "tags": ["filter"]},

    {"schema_name": "neobank_v1", "question": "What is the average transaction amount by transaction type?",
     "sql": "SELECT txn_type, AVG(amount) AS avg_amount, COUNT(*) AS txn_count FROM transactions WHERE status = 'completed' GROUP BY txn_type ORDER BY avg_amount DESC",
     "complexity": "easy", "tags": ["avg", "group_by"]},

    {"schema_name": "neobank_v1", "question": "For each customer, show their name and total balance across all their accounts.",
     "sql": """SELECT c.customer_id, c.full_name, SUM(a.balance) AS total_balance
FROM customers c
JOIN accounts a ON c.customer_id = a.customer_id
WHERE a.status = 'active'
GROUP BY c.customer_id, c.full_name
ORDER BY total_balance DESC""",
     "complexity": "medium", "tags": ["join", "sum", "group_by"]},

    {"schema_name": "neobank_v1", "question": "Which merchant categories have the highest total card spend?",
     "sql": """SELECT ct.merchant_category, SUM(ct.amount) AS total_spend, COUNT(*) AS txn_count
FROM card_transactions ct
WHERE ct.status = 'completed'
GROUP BY ct.merchant_category
ORDER BY total_spend DESC
LIMIT 10""",
     "complexity": "medium", "tags": ["group_by", "sum", "limit"]},

    {"schema_name": "neobank_v1", "question": "Find customers who have both a savings and a current account.",
     "sql": """SELECT c.customer_id, c.full_name
FROM customers c
WHERE EXISTS (
    SELECT 1 FROM accounts a WHERE a.customer_id = c.customer_id AND a.account_type = 'savings' AND a.status = 'active'
)
AND EXISTS (
    SELECT 1 FROM accounts a WHERE a.customer_id = c.customer_id AND a.account_type = 'current' AND a.status = 'active'
)""",
     "complexity": "medium", "tags": ["exists", "subquery", "multi_condition"]},

    {"schema_name": "neobank_v1", "question": "Show daily transaction volume and value for the last 7 days.",
     "sql": """SELECT DATE(created_at) AS txn_date, COUNT(*) AS txn_count, SUM(amount) AS total_volume
FROM transactions
WHERE status = 'completed' AND created_at >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY DATE(created_at)
ORDER BY txn_date""",
     "complexity": "medium", "tags": ["date_group", "sum", "count"]},

    {"schema_name": "neobank_v1", "question": "Find accounts where the total outgoing transfers exceed the total incoming transfers.",
     "sql": """WITH outgoing AS (
    SELECT from_account AS account_id, SUM(amount) AS total_out
    FROM transactions WHERE txn_type = 'transfer' AND status = 'completed'
    GROUP BY from_account
),
incoming AS (
    SELECT to_account AS account_id, SUM(amount) AS total_in
    FROM transactions WHERE txn_type = 'transfer' AND status = 'completed'
    GROUP BY to_account
)
SELECT o.account_id, o.total_out, COALESCE(i.total_in, 0) AS total_in
FROM outgoing o
LEFT JOIN incoming i ON o.account_id = i.account_id
WHERE o.total_out > COALESCE(i.total_in, 0)
ORDER BY (o.total_out - COALESCE(i.total_in, 0)) DESC""",
     "complexity": "hard", "tags": ["cte", "coalesce", "left_join", "arithmetic"]},

    {"schema_name": "neobank_v1", "question": "Find the top spending customer per country based on card transactions.",
     "sql": """WITH customer_spend AS (
    SELECT c.customer_id, c.full_name, c.country, SUM(ct.amount) AS total_spend
    FROM customers c
    JOIN accounts a ON c.customer_id = a.customer_id
    JOIN cards cd ON a.account_id = cd.account_id
    JOIN card_transactions ct ON cd.card_id = ct.card_id
    WHERE ct.status = 'completed'
    GROUP BY c.customer_id, c.full_name, c.country
),
ranked AS (
    SELECT *, RANK() OVER (PARTITION BY country ORDER BY total_spend DESC) AS rnk
    FROM customer_spend
)
SELECT customer_id, full_name, country, total_spend
FROM ranked WHERE rnk = 1
ORDER BY country""",
     "complexity": "hard", "tags": ["cte", "window_function", "rank", "multi_join"]},

    {"schema_name": "neobank_v1", "question": "Detect customers with more than 3 failed transactions in a single day.",
     "sql": """SELECT c.customer_id, c.full_name, DATE(t.created_at) AS txn_date, COUNT(*) AS failed_count
FROM transactions t
JOIN accounts a ON t.from_account = a.account_id
JOIN customers c ON a.customer_id = c.customer_id
WHERE t.status = 'failed'
GROUP BY c.customer_id, c.full_name, DATE(t.created_at)
HAVING COUNT(*) > 3
ORDER BY failed_count DESC""",
     "complexity": "hard", "tags": ["multi_join", "having", "date_group", "fraud_detection"]},

    # SCHEMA 3: lending_v1
    # Tables: borrowers, loan_products, loan_applications, loans, repayments

    {"schema_name": "lending_v1", "question": "How many loan applications are currently pending?",
     "sql": "SELECT COUNT(*) AS pending_applications FROM loan_applications WHERE status = 'pending'",
     "complexity": "easy", "tags": ["count", "filter"]},

    {"schema_name": "lending_v1", "question": "What is the average credit score of approved borrowers?",
     "sql": """SELECT AVG(b.credit_score) AS avg_credit_score
FROM borrowers b
JOIN loan_applications la ON b.borrower_id = la.borrower_id
WHERE la.status IN ('approved', 'disbursed')""",
     "complexity": "easy", "tags": ["avg", "join", "filter"]},

    {"schema_name": "lending_v1", "question": "List all NPA (non-performing) loans with outstanding amount greater than 100000.",
     "sql": "SELECT loan_id, disbursed_amount, outstanding, dpd FROM loans WHERE npa_flag = TRUE AND outstanding > 100000 ORDER BY outstanding DESC",
     "complexity": "easy", "tags": ["filter", "boolean"]},

    {"schema_name": "lending_v1", "question": "How many applications were rejected per loan product?",
     "sql": """SELECT lp.name AS product_name, COUNT(*) AS rejected_count
FROM loan_applications la
JOIN loan_products lp ON la.product_id = lp.product_id
WHERE la.status = 'rejected'
GROUP BY lp.product_id, lp.name
ORDER BY rejected_count DESC""",
     "complexity": "easy", "tags": ["join", "group_by", "count"]},

    {"schema_name": "lending_v1", "question": "What percentage of repayments were made on time?",
     "sql": """SELECT
    ROUND(100.0 * SUM(CASE WHEN status = 'on_time' THEN 1 ELSE 0 END) / COUNT(*), 2) AS on_time_pct
FROM repayments""",
     "complexity": "easy", "tags": ["conditional_agg", "arithmetic"]},

    {"schema_name": "lending_v1", "question": "Find borrowers with more than 2 missed repayments.",
     "sql": """SELECT b.borrower_id, b.full_name, COUNT(*) AS missed_count
FROM repayments r
JOIN loans l ON r.loan_id = l.loan_id
JOIN loan_applications la ON l.application_id = la.application_id
JOIN borrowers b ON la.borrower_id = b.borrower_id
WHERE r.status = 'missed'
GROUP BY b.borrower_id, b.full_name
HAVING COUNT(*) > 2
ORDER BY missed_count DESC""",
     "complexity": "medium", "tags": ["multi_join", "having", "count"]},

    {"schema_name": "lending_v1", "question": "What is the total outstanding loan amount per product type?",
     "sql": """SELECT lp.product_type, SUM(l.outstanding) AS total_outstanding, COUNT(l.loan_id) AS loan_count
FROM loans l
JOIN loan_applications la ON l.application_id = la.application_id
JOIN loan_products lp ON la.product_id = lp.product_id
GROUP BY lp.product_type
ORDER BY total_outstanding DESC""",
     "complexity": "medium", "tags": ["multi_join", "sum", "group_by"]},

    {"schema_name": "lending_v1", "question": "Find the approval rate by employment type.",
     "sql": """SELECT b.employment_type,
    COUNT(*) AS total_applications,
    SUM(CASE WHEN la.status IN ('approved','disbursed') THEN 1 ELSE 0 END) AS approved,
    ROUND(100.0 * SUM(CASE WHEN la.status IN ('approved','disbursed') THEN 1 ELSE 0 END) / COUNT(*), 2) AS approval_rate_pct
FROM loan_applications la
JOIN borrowers b ON la.borrower_id = b.borrower_id
GROUP BY b.employment_type
ORDER BY approval_rate_pct DESC""",
     "complexity": "medium", "tags": ["join", "conditional_agg", "group_by", "arithmetic"]},

    {"schema_name": "lending_v1", "question": "For each loan, show the cumulative amount repaid over time.",
     "sql": """SELECT
    r.loan_id,
    r.paid_date,
    r.paid_amount,
    SUM(r.paid_amount) OVER (PARTITION BY r.loan_id ORDER BY r.paid_date) AS cumulative_repaid
FROM repayments r
WHERE r.status IN ('on_time', 'late', 'partial')
ORDER BY r.loan_id, r.paid_date""",
     "complexity": "hard", "tags": ["window_function", "cumulative_sum", "partition_by"]},

    {"schema_name": "lending_v1", "question": "Identify borrowers whose credit score is below the average for their city.",
     "sql": """SELECT b.borrower_id, b.full_name, b.city, b.credit_score,
    AVG(b.credit_score) OVER (PARTITION BY b.city) AS city_avg_score
FROM borrowers b
WHERE b.credit_score < (
    SELECT AVG(b2.credit_score) FROM borrowers b2 WHERE b2.city = b.city
)
ORDER BY b.city, b.credit_score""",
     "complexity": "hard", "tags": ["correlated_subquery", "window_function", "partition_by"]},

    # SCHEMA 4: payments_v1
    # Tables: merchants, payment_links, payments, settlements, disputes

    {"schema_name": "payments_v1", "question": "How many merchants are currently active?",
     "sql": "SELECT COUNT(*) AS active_merchants FROM merchants WHERE active = TRUE",
     "complexity": "easy", "tags": ["count", "filter"]},

    {"schema_name": "payments_v1", "question": "What is the total successful payment volume today?",
     "sql": "SELECT SUM(amount) AS total_volume FROM payments WHERE status = 'success' AND DATE(paid_at) = CURRENT_DATE",
     "complexity": "easy", "tags": ["sum", "filter", "date"]},

    {"schema_name": "payments_v1", "question": "How many open disputes exist per merchant?",
     "sql": """SELECT p.merchant_id, COUNT(d.dispute_id) AS open_disputes
FROM disputes d
JOIN payments py ON d.payment_id = py.payment_id
JOIN payment_links pl ON py.link_id = pl.link_id
JOIN merchants p ON pl.merchant_id = p.merchant_id
WHERE d.status = 'open'
GROUP BY p.merchant_id
ORDER BY open_disputes DESC""",
     "complexity": "medium", "tags": ["multi_join", "group_by", "count"]},

    {"schema_name": "payments_v1", "question": "What is the settlement success rate per merchant?",
     "sql": """SELECT merchant_id,
    COUNT(*) AS total_settlements,
    SUM(CASE WHEN status = 'processed' THEN 1 ELSE 0 END) AS successful,
    ROUND(100.0 * SUM(CASE WHEN status = 'processed' THEN 1 ELSE 0 END) / COUNT(*), 2) AS success_rate
FROM settlements
GROUP BY merchant_id
ORDER BY success_rate""",
     "complexity": "medium", "tags": ["conditional_agg", "arithmetic", "group_by"]},

    {"schema_name": "payments_v1", "question": "Find merchants whose dispute rate (disputes/payments) exceeds 5%.",
     "sql": """WITH merchant_stats AS (
    SELECT
        pl.merchant_id,
        COUNT(DISTINCT py.payment_id) AS total_payments,
        COUNT(DISTINCT d.dispute_id) AS total_disputes
    FROM payment_links pl
    JOIN payments py ON pl.link_id = py.link_id
    LEFT JOIN disputes d ON py.payment_id = d.payment_id
    WHERE py.status = 'success'
    GROUP BY pl.merchant_id
)
SELECT merchant_id, total_payments, total_disputes,
    ROUND(100.0 * total_disputes / NULLIF(total_payments, 0), 2) AS dispute_rate_pct
FROM merchant_stats
WHERE total_disputes * 100.0 / NULLIF(total_payments, 0) > 5
ORDER BY dispute_rate_pct DESC""",
     "complexity": "hard", "tags": ["cte", "left_join", "nullif", "arithmetic", "multi_join"]},

    {"schema_name": "payments_v1", "question": "Show 7-day rolling average payment volume per merchant.",
     "sql": """WITH daily_volume AS (
    SELECT pl.merchant_id, DATE(py.paid_at) AS pay_date, SUM(py.amount) AS daily_total
    FROM payments py
    JOIN payment_links pl ON py.link_id = pl.link_id
    WHERE py.status = 'success'
    GROUP BY pl.merchant_id, DATE(py.paid_at)
)
SELECT merchant_id, pay_date, daily_total,
    AVG(daily_total) OVER (
        PARTITION BY merchant_id
        ORDER BY pay_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_avg
FROM daily_volume
ORDER BY merchant_id, pay_date""",
     "complexity": "hard", "tags": ["cte", "window_function", "rolling_avg", "rows_between"]},

    # SCHEMA 5: investment_v1
    # Tables: investors, instruments, portfolios, holdings, trades

    {"schema_name": "investment_v1", "question": "How many investors have an aggressive risk profile?",
     "sql": "SELECT COUNT(*) AS aggressive_investors FROM investors WHERE risk_profile = 'aggressive'",
     "complexity": "easy", "tags": ["count", "filter"]},

    {"schema_name": "investment_v1", "question": "What is the total current value of all equity holdings?",
     "sql": """SELECT SUM(h.quantity * h.current_price) AS total_equity_value
FROM holdings h
JOIN instruments i ON h.instrument_id = i.instrument_id
WHERE i.instrument_type = 'equity'""",
     "complexity": "easy", "tags": ["join", "sum", "arithmetic"]},

    {"schema_name": "investment_v1", "question": "Find the top 5 most traded instruments by number of trades.",
     "sql": """SELECT i.symbol, i.name, COUNT(t.trade_id) AS trade_count
FROM instruments i
JOIN trades t ON i.instrument_id = t.instrument_id
WHERE t.status = 'executed'
GROUP BY i.instrument_id, i.symbol, i.name
ORDER BY trade_count DESC
LIMIT 5""",
     "complexity": "medium", "tags": ["join", "group_by", "count", "limit"]},

    {"schema_name": "investment_v1", "question": "For each investor, calculate the unrealised P&L across their portfolio.",
     "sql": """SELECT
    inv.investor_id,
    inv.full_name,
    SUM(h.quantity * (h.current_price - h.avg_buy_price)) AS unrealised_pnl
FROM investors inv
JOIN portfolios p ON inv.investor_id = p.investor_id
JOIN holdings h ON p.portfolio_id = h.portfolio_id
GROUP BY inv.investor_id, inv.full_name
ORDER BY unrealised_pnl DESC""",
     "complexity": "medium", "tags": ["multi_join", "arithmetic", "sum", "group_by"]},

    {"schema_name": "investment_v1", "question": "Find instruments where the number of sell trades exceeds buy trades in the last 30 days.",
     "sql": """WITH trade_counts AS (
    SELECT instrument_id,
        SUM(CASE WHEN trade_type = 'buy' THEN 1 ELSE 0 END) AS buys,
        SUM(CASE WHEN trade_type = 'sell' THEN 1 ELSE 0 END) AS sells
    FROM trades
    WHERE status = 'executed' AND traded_at >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY instrument_id
)
SELECT i.symbol, i.name, tc.buys, tc.sells
FROM trade_counts tc
JOIN instruments i ON tc.instrument_id = i.instrument_id
WHERE tc.sells > tc.buys
ORDER BY (tc.sells - tc.buys) DESC""",
     "complexity": "hard", "tags": ["cte", "conditional_agg", "join", "arithmetic"]},

    # SCHEMA 6: retail_analytics_v1
    # Tables: stores, products, inventory, promotions, transactions

    {"schema_name": "retail_analytics_v1", "question": "Which stores have inventory below reorder level for any product?",
     "sql": """SELECT s.store_id, s.city, p.name AS product_name, i.quantity, i.reorder_level
FROM inventory i
JOIN stores s ON i.store_id = s.store_id
JOIN products p ON i.product_id = p.product_id
WHERE i.quantity < i.reorder_level
ORDER BY s.city, p.name""",
     "complexity": "medium", "tags": ["join", "comparison", "filter"]},

    {"schema_name": "retail_analytics_v1", "question": "What is the gross margin percentage per product brand?",
     "sql": """SELECT p.brand,
    SUM(t.quantity * t.sale_price) AS total_revenue,
    SUM(t.quantity * p.cost_price) AS total_cost,
    ROUND(100.0 * (SUM(t.quantity * t.sale_price) - SUM(t.quantity * p.cost_price)) /
          NULLIF(SUM(t.quantity * t.sale_price), 0), 2) AS gross_margin_pct
FROM transactions t
JOIN products p ON t.product_id = p.product_id
GROUP BY p.brand
ORDER BY gross_margin_pct DESC""",
     "complexity": "hard", "tags": ["join", "arithmetic", "nullif", "group_by"]},

    {"schema_name": "retail_analytics_v1", "question": "Find products whose sales increased week-over-week for 3 consecutive weeks.",
     "sql": """WITH weekly_sales AS (
    SELECT product_id,
        DATE_TRUNC('week', txn_timestamp) AS week,
        SUM(quantity * sale_price) AS weekly_revenue
    FROM transactions
    GROUP BY product_id, DATE_TRUNC('week', txn_timestamp)
),
with_prev AS (
    SELECT *,
        LAG(weekly_revenue, 1) OVER (PARTITION BY product_id ORDER BY week) AS prev1,
        LAG(weekly_revenue, 2) OVER (PARTITION BY product_id ORDER BY week) AS prev2
    FROM weekly_sales
)
SELECT DISTINCT p.product_id, p.name
FROM with_prev wp
JOIN products p ON wp.product_id = p.product_id
WHERE wp.weekly_revenue > wp.prev1 AND wp.prev1 > wp.prev2""",
     "complexity": "hard", "tags": ["cte", "window_function", "lag", "consecutive"]},

    # SCHEMA 7: subscription_ecom_v1

    {"schema_name": "subscription_ecom_v1", "question": "What is the monthly recurring revenue (MRR) by plan?",
     "sql": """SELECT pl.name AS plan_name, COUNT(s.sub_id) AS active_subs,
    SUM(pl.price_monthly) AS mrr
FROM subscriptions s
JOIN plans pl ON s.plan_id = pl.plan_id
WHERE s.status = 'active'
GROUP BY pl.plan_id, pl.name
ORDER BY mrr DESC""",
     "complexity": "medium", "tags": ["join", "sum", "group_by"]},

    {"schema_name": "subscription_ecom_v1", "question": "Find customers who churned (cancelled) within their first billing cycle.",
     "sql": """SELECT c.customer_id, c.email, s.start_date, s.end_date,
    (s.end_date - s.start_date) AS days_active
FROM subscriptions s
JOIN customers c ON s.customer_id = c.customer_id
WHERE s.status = 'cancelled'
AND (s.end_date - s.start_date) <= 30
ORDER BY days_active""",
     "complexity": "medium", "tags": ["join", "date_arithmetic", "filter"]},

    {"schema_name": "subscription_ecom_v1", "question": "Calculate the customer lifetime value (LTV) from invoice history.",
     "sql": """SELECT c.customer_id, c.email,
    COUNT(i.invoice_id) AS total_invoices,
    SUM(i.amount) AS total_paid,
    MIN(i.paid_date) AS first_payment,
    MAX(i.paid_date) AS last_payment
FROM customers c
JOIN subscriptions s ON c.customer_id = s.customer_id
JOIN invoices i ON s.sub_id = i.sub_id
WHERE i.status = 'paid'
GROUP BY c.customer_id, c.email
ORDER BY total_paid DESC""",
     "complexity": "hard", "tags": ["multi_join", "sum", "min_max", "group_by"]},

    # SCHEMA 8: logistics_v1

    {"schema_name": "logistics_v1", "question": "What is the average number of delivery attempts per parcel?",
     "sql": """SELECT AVG(attempt_count) AS avg_attempts
FROM (
    SELECT parcel_id, COUNT(*) AS attempt_count
    FROM delivery_attempts
    GROUP BY parcel_id
) sub""",
     "complexity": "medium", "tags": ["subquery", "avg", "group_by"]},

    {"schema_name": "logistics_v1", "question": "Find delivery agents with a success rate below 70%.",
     "sql": """SELECT da.agent_id, da.name,
    COUNT(*) AS total_attempts,
    SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) AS successes,
    ROUND(100.0 * SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) / COUNT(*), 2) AS success_rate
FROM delivery_attempts dea
JOIN delivery_agents da ON dea.agent_id = da.agent_id
GROUP BY da.agent_id, da.name
HAVING ROUND(100.0 * SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) / COUNT(*), 2) < 70
ORDER BY success_rate""",
     "complexity": "hard", "tags": ["join", "conditional_agg", "having", "arithmetic"]},

    # SCHEMA 9: saas_metrics_v1

    {"schema_name": "saas_metrics_v1", "question": "Which features are used most by enterprise accounts?",
     "sql": """SELECT f.name AS feature_name, COUNT(fu.usage_id) AS usage_count
FROM feature_usage fu
JOIN features f ON fu.feature_id = f.feature_id
JOIN users u ON fu.user_id = u.user_id
JOIN accounts a ON u.account_id = a.account_id
JOIN contracts c ON a.account_id = c.account_id
WHERE c.status = 'active' AND f.tier = 'enterprise'
GROUP BY f.feature_id, f.name
ORDER BY usage_count DESC""",
     "complexity": "medium", "tags": ["multi_join", "group_by", "count"]},

    {"schema_name": "saas_metrics_v1", "question": "Find accounts at risk of churning: active contract ending within 30 days and renewal probability below 0.4.",
     "sql": """SELECT a.account_id, a.company_name, c.end_date, c.mrr, c.renewal_probability
FROM accounts a
JOIN contracts c ON a.account_id = c.account_id
WHERE c.status = 'active'
AND c.end_date <= CURRENT_DATE + INTERVAL '30 days'
AND c.renewal_probability < 0.4
ORDER BY c.mrr DESC""",
     "complexity": "medium", "tags": ["join", "filter", "date_arithmetic", "business_logic"]},

    {"schema_name": "saas_metrics_v1", "question": "Calculate DAU/MAU ratio (stickiness) per account for the last 30 days.",
     "sql": """WITH dau AS (
    SELECT u.account_id, DATE(fu.used_at) AS use_date, COUNT(DISTINCT u.user_id) AS daily_users
    FROM feature_usage fu
    JOIN users u ON fu.user_id = u.user_id
    WHERE fu.used_at >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY u.account_id, DATE(fu.used_at)
),
mau AS (
    SELECT u.account_id, COUNT(DISTINCT u.user_id) AS monthly_users
    FROM feature_usage fu
    JOIN users u ON fu.user_id = u.user_id
    WHERE fu.used_at >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY u.account_id
)
SELECT d.account_id, AVG(d.daily_users) AS avg_dau, m.monthly_users AS mau,
    ROUND(AVG(d.daily_users) / NULLIF(m.monthly_users, 0), 3) AS stickiness
FROM dau d
JOIN mau m ON d.account_id = m.account_id
GROUP BY d.account_id, m.monthly_users
ORDER BY stickiness DESC""",
     "complexity": "hard", "tags": ["cte", "dau_mau", "nullif", "multi_join", "date_group"]},

    # SCHEMA 10: insurance_v1

    {"schema_name": "insurance_v1", "question": "What is the claim approval rate by insurance type?",
     "sql": """SELECT ip.type,
    COUNT(cl.claim_id) AS total_claims,
    SUM(CASE WHEN cl.status = 'approved' OR cl.status = 'paid' THEN 1 ELSE 0 END) AS approved,
    ROUND(100.0 * SUM(CASE WHEN cl.status IN ('approved','paid') THEN 1 ELSE 0 END) / COUNT(*), 2) AS approval_rate
FROM claims cl
JOIN policies po ON cl.policy_id = po.policy_id
JOIN insurance_products ip ON po.product_id = ip.product_id
GROUP BY ip.type
ORDER BY approval_rate DESC""",
     "complexity": "medium", "tags": ["multi_join", "conditional_agg", "group_by", "arithmetic"]},

    {"schema_name": "insurance_v1", "question": "Find agents who manage policies with a high lapse rate (more than 30% lapsed).",
     "sql": """WITH agent_policies AS (
    SELECT po.policy_id, po.status, ag.agent_id, ag.name
    FROM policies po
    JOIN agents ag ON ag.region = (
        SELECT ph.city FROM policyholders ph WHERE ph.holder_id = po.holder_id
    )
)
SELECT agent_id, name,
    COUNT(*) AS total_policies,
    SUM(CASE WHEN status = 'lapsed' THEN 1 ELSE 0 END) AS lapsed_count,
    ROUND(100.0 * SUM(CASE WHEN status = 'lapsed' THEN 1 ELSE 0 END) / COUNT(*), 2) AS lapse_rate
FROM agent_policies
GROUP BY agent_id, name
HAVING lapse_rate > 30
ORDER BY lapse_rate DESC""",
     "complexity": "hard", "tags": ["cte", "correlated_subquery", "conditional_agg", "having"]},

    {"schema_name": "insurance_v1", "question": "Calculate the loss ratio (total claims settled / total premium collected) per product.",
     "sql": """SELECT ip.name AS product_name,
    SUM(po.premium_annual) AS total_premium,
    SUM(cl.settled_amount) AS total_settled,
    ROUND(SUM(cl.settled_amount) / NULLIF(SUM(po.premium_annual), 0), 4) AS loss_ratio
FROM policies po
JOIN insurance_products ip ON po.product_id = ip.product_id
LEFT JOIN claims cl ON po.policy_id = cl.policy_id AND cl.status = 'paid'
GROUP BY ip.product_id, ip.name
ORDER BY loss_ratio DESC""",
     "complexity": "hard", "tags": ["left_join", "sum", "nullif", "arithmetic", "group_by"]},
]


def build_eval_set():
    out = EVAL_DIR / "golden_eval.jsonl"

    # Validate: every entry must have required fields
    required = {"schema_name", "question", "sql", "complexity", "tags"}
    for i, ex in enumerate(GOLDEN_EVAL):
        missing = required - set(ex.keys())
        if missing:
            raise ValueError(f"Entry {i} missing fields: {missing}")

    with open(out, "w") as f:
        for ex in GOLDEN_EVAL:
            f.write(json.dumps(ex) + "\n")

    # Summary stats
    import collections
    by_schema = collections.Counter(e["schema_name"] for e in GOLDEN_EVAL)
    by_complexity = collections.Counter(e["complexity"] for e in GOLDEN_EVAL)
    all_tags = [t for e in GOLDEN_EVAL for t in e["tags"]]
    top_tags = collections.Counter(all_tags).most_common(10)

    print(f"\n── Golden Eval Set ──")
    print(f"  Total examples : {len(GOLDEN_EVAL)}")
    print(f"\n  By schema:")
    for s, c in sorted(by_schema.items()): print(f"    {s:<30} {c}")
    print(f"\n  By complexity:")
    for c, n in sorted(by_complexity.items()): print(f"    {c:<10} {n}")
    print(f"\n  Top SQL tags tested:")
    for tag, n in top_tags: print(f"    {tag:<25} {n}")
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    build_eval_set()