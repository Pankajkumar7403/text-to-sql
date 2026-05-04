"""
Day 1 Script: Generate 10 synthetic schemas (5 e-commerce, 5 fintech).
These become the base for Layer 2 synthetic data generation (Day 3).
Output: data/raw/schemas/*.json
"""

import json
from pathlib import Path

ROOT   = Path(__file__).parent.parent
SCHEMAS = ROOT / "data" / "raw" / "schemas"
SCHEMAS.mkdir(parents=True, exist_ok=True)


ECOMMERCE_SCHEMAS = [

    {
        "name": "marketplace_v1",
        "domain": "ecommerce",
        "description": "Multi-vendor marketplace with sellers, products, orders",
        "tables": {
            "users": {
                "columns": ["user_id INT PK", "email VARCHAR", "full_name VARCHAR",
                            "created_at TIMESTAMP", "country VARCHAR", "is_active BOOL"],
                "sample_values": {"country": ["IN", "US", "UK"], "is_active": [True, False]}
            },
            "sellers": {
                "columns": ["seller_id INT PK", "user_id INT FK(users)", "shop_name VARCHAR",
                            "rating DECIMAL(3,2)", "total_sales INT", "joined_at TIMESTAMP"],
            },
            "categories": {
                "columns": ["category_id INT PK", "name VARCHAR", "parent_id INT FK(categories)"],
            },
            "products": {
                "columns": ["product_id INT PK", "seller_id INT FK(sellers)",
                            "category_id INT FK(categories)", "title VARCHAR",
                            "price DECIMAL(10,2)", "stock_qty INT",
                            "created_at TIMESTAMP", "is_listed BOOL"],
            },
            "orders": {
                "columns": ["order_id INT PK", "user_id INT FK(users)",
                            "status VARCHAR", "total_amount DECIMAL(10,2)",
                            "created_at TIMESTAMP", "shipped_at TIMESTAMP"],
                "sample_values": {"status": ["pending", "shipped", "delivered", "cancelled"]}
            },
            "order_items": {
                "columns": ["item_id INT PK", "order_id INT FK(orders)",
                            "product_id INT FK(products)", "quantity INT",
                            "unit_price DECIMAL(10,2)"],
            },
            "reviews": {
                "columns": ["review_id INT PK", "product_id INT FK(products)",
                            "user_id INT FK(users)", "rating INT",
                            "body TEXT", "created_at TIMESTAMP"],
            },
        }
    },

    {
        "name": "subscription_ecom_v1",
        "domain": "ecommerce",
        "description": "Subscription-box e-commerce with recurring billing",
        "tables": {
            "customers": {
                "columns": ["customer_id INT PK", "email VARCHAR", "signup_date DATE",
                            "referral_source VARCHAR", "ltv DECIMAL(10,2)"],
            },
            "plans": {
                "columns": ["plan_id INT PK", "name VARCHAR", "price_monthly DECIMAL(8,2)",
                            "billing_cycle VARCHAR", "max_boxes INT"],
                "sample_values": {"billing_cycle": ["monthly", "quarterly", "annual"]}
            },
            "subscriptions": {
                "columns": ["sub_id INT PK", "customer_id INT FK(customers)",
                            "plan_id INT FK(plans)", "status VARCHAR",
                            "start_date DATE", "end_date DATE", "next_billing DATE"],
                "sample_values": {"status": ["active", "paused", "cancelled", "past_due"]}
            },
            "shipments": {
                "columns": ["shipment_id INT PK", "sub_id INT FK(subscriptions)",
                            "shipped_date DATE", "delivered_date DATE",
                            "carrier VARCHAR", "tracking_number VARCHAR"],
            },
            "invoices": {
                "columns": ["invoice_id INT PK", "sub_id INT FK(subscriptions)",
                            "amount DECIMAL(8,2)", "status VARCHAR",
                            "issued_date DATE", "paid_date DATE"],
                "sample_values": {"status": ["paid", "failed", "pending", "refunded"]}
            },
        }
    },

    {
        "name": "retail_analytics_v1",
        "domain": "ecommerce",
        "description": "Retail store with inventory, promotions, and POS data",
        "tables": {
            "stores": {
                "columns": ["store_id INT PK", "city VARCHAR", "state VARCHAR",
                            "opened_date DATE", "sq_footage INT", "manager_id INT"],
            },
            "products": {
                "columns": ["product_id INT PK", "sku VARCHAR", "name VARCHAR",
                            "brand VARCHAR", "category VARCHAR",
                            "cost_price DECIMAL(8,2)", "retail_price DECIMAL(8,2)"],
            },
            "inventory": {
                "columns": ["inv_id INT PK", "store_id INT FK(stores)",
                            "product_id INT FK(products)", "quantity INT",
                            "reorder_level INT", "last_restocked DATE"],
            },
            "promotions": {
                "columns": ["promo_id INT PK", "name VARCHAR", "discount_pct DECIMAL(4,2)",
                            "start_date DATE", "end_date DATE", "promo_type VARCHAR"],
                "sample_values": {"promo_type": ["bogo", "flash_sale", "seasonal", "clearance"]}
            },
            "transactions": {
                "columns": ["txn_id INT PK", "store_id INT FK(stores)",
                            "product_id INT FK(products)", "promo_id INT FK(promotions)",
                            "quantity INT", "sale_price DECIMAL(8,2)", "txn_timestamp TIMESTAMP"],
            },
        }
    },

    {
        "name": "logistics_v1",
        "domain": "ecommerce",
        "description": "Last-mile delivery and logistics operations",
        "tables": {
            "warehouses": {
                "columns": ["warehouse_id INT PK", "city VARCHAR", "capacity INT",
                            "current_load INT", "region VARCHAR"],
            },
            "delivery_agents": {
                "columns": ["agent_id INT PK", "name VARCHAR", "warehouse_id INT FK(warehouses)",
                            "vehicle_type VARCHAR", "rating DECIMAL(3,2)", "active BOOL"],
            },
            "parcels": {
                "columns": ["parcel_id INT PK", "sender_id INT", "receiver_city VARCHAR",
                            "weight_kg DECIMAL(6,2)", "declared_value DECIMAL(10,2)",
                            "created_at TIMESTAMP", "status VARCHAR"],
                "sample_values": {"status": ["picked_up", "in_transit", "out_for_delivery", "delivered", "failed"]}
            },
            "delivery_attempts": {
                "columns": ["attempt_id INT PK", "parcel_id INT FK(parcels)",
                            "agent_id INT FK(delivery_agents)", "attempt_time TIMESTAMP",
                            "outcome VARCHAR", "failure_reason VARCHAR"],
                "sample_values": {"outcome": ["success", "failed", "rescheduled"]}
            },
            "routes": {
                "columns": ["route_id INT PK", "warehouse_id INT FK(warehouses)",
                            "agent_id INT FK(delivery_agents)", "date DATE",
                            "total_stops INT", "completed_stops INT", "distance_km DECIMAL(6,1)"],
            },
        }
    },

    {
        "name": "saas_metrics_v1",
        "domain": "ecommerce",
        "description": "SaaS product usage and revenue metrics",
        "tables": {
            "accounts": {
                "columns": ["account_id INT PK", "company_name VARCHAR", "industry VARCHAR",
                            "employee_count INT", "arr DECIMAL(12,2)", "created_at DATE"],
            },
            "users": {
                "columns": ["user_id INT PK", "account_id INT FK(accounts)", "email VARCHAR",
                            "role VARCHAR", "last_login TIMESTAMP", "is_admin BOOL"],
            },
            "features": {
                "columns": ["feature_id INT PK", "name VARCHAR", "tier VARCHAR",
                            "release_date DATE"],
                "sample_values": {"tier": ["free", "pro", "enterprise"]}
            },
            "feature_usage": {
                "columns": ["usage_id INT PK", "user_id INT FK(users)",
                            "feature_id INT FK(features)", "used_at TIMESTAMP",
                            "session_seconds INT"],
            },
            "contracts": {
                "columns": ["contract_id INT PK", "account_id INT FK(accounts)",
                            "start_date DATE", "end_date DATE", "mrr DECIMAL(10,2)",
                            "status VARCHAR", "renewal_probability DECIMAL(4,2)"],
                "sample_values": {"status": ["active", "at_risk", "churned", "renewed"]}
            },
        }
    },

]

FINTECH_SCHEMAS = [

    {
        "name": "neobank_v1",
        "domain": "fintech",
        "description": "Digital bank with accounts, transactions, and cards",
        "tables": {
            "customers": {
                "columns": ["customer_id INT PK", "full_name VARCHAR", "email VARCHAR",
                            "kyc_status VARCHAR", "risk_tier VARCHAR",
                            "created_at TIMESTAMP", "country VARCHAR"],
                "sample_values": {"kyc_status": ["pending", "verified", "rejected"],
                                  "risk_tier": ["low", "medium", "high"]}
            },
            "accounts": {
                "columns": ["account_id INT PK", "customer_id INT FK(customers)",
                            "account_type VARCHAR", "balance DECIMAL(14,2)",
                            "currency VARCHAR", "status VARCHAR", "opened_date DATE"],
                "sample_values": {"account_type": ["savings", "current", "wallet"],
                                  "status": ["active", "frozen", "closed"]}
            },
            "transactions": {
                "columns": ["txn_id INT PK", "from_account INT FK(accounts)",
                            "to_account INT FK(accounts)", "amount DECIMAL(14,2)",
                            "txn_type VARCHAR", "status VARCHAR",
                            "created_at TIMESTAMP", "reference VARCHAR"],
                "sample_values": {"txn_type": ["transfer", "deposit", "withdrawal", "refund"],
                                  "status": ["completed", "pending", "failed", "reversed"]}
            },
            "cards": {
                "columns": ["card_id INT PK", "account_id INT FK(accounts)",
                            "card_type VARCHAR", "last4 CHAR(4)",
                            "expiry_date DATE", "status VARCHAR"],
                "sample_values": {"card_type": ["debit", "prepaid"],
                                  "status": ["active", "blocked", "expired"]}
            },
            "card_transactions": {
                "columns": ["card_txn_id INT PK", "card_id INT FK(cards)",
                            "merchant_name VARCHAR", "merchant_category VARCHAR",
                            "amount DECIMAL(10,2)", "currency VARCHAR",
                            "status VARCHAR", "txn_at TIMESTAMP"],
            },
        }
    },

    {
        "name": "lending_v1",
        "domain": "fintech",
        "description": "Consumer lending with loans, repayments, and credit scoring",
        "tables": {
            "borrowers": {
                "columns": ["borrower_id INT PK", "full_name VARCHAR", "dob DATE",
                            "credit_score INT", "annual_income DECIMAL(12,2)",
                            "employment_type VARCHAR", "city VARCHAR"],
                "sample_values": {"employment_type": ["salaried", "self_employed", "gig", "unemployed"]}
            },
            "loan_products": {
                "columns": ["product_id INT PK", "name VARCHAR", "min_amount DECIMAL(10,2)",
                            "max_amount DECIMAL(10,2)", "interest_rate_pct DECIMAL(4,2)",
                            "max_tenure_months INT", "product_type VARCHAR"],
                "sample_values": {"product_type": ["personal", "home", "auto", "bnpl", "education"]}
            },
            "loan_applications": {
                "columns": ["application_id INT PK", "borrower_id INT FK(borrowers)",
                            "product_id INT FK(loan_products)", "requested_amount DECIMAL(12,2)",
                            "requested_tenure INT", "status VARCHAR", "applied_at TIMESTAMP",
                            "decision_at TIMESTAMP", "rejection_reason VARCHAR"],
                "sample_values": {"status": ["pending", "approved", "rejected", "disbursed"]}
            },
            "loans": {
                "columns": ["loan_id INT PK", "application_id INT FK(loan_applications)",
                            "disbursed_amount DECIMAL(12,2)", "disbursed_at DATE",
                            "emi_amount DECIMAL(10,2)", "outstanding DECIMAL(12,2)",
                            "dpd INT", "npa_flag BOOL"],
            },
            "repayments": {
                "columns": ["repayment_id INT PK", "loan_id INT FK(loans)",
                            "due_date DATE", "paid_date DATE",
                            "due_amount DECIMAL(10,2)", "paid_amount DECIMAL(10,2)",
                            "payment_mode VARCHAR", "status VARCHAR"],
                "sample_values": {"status": ["on_time", "late", "partial", "missed"],
                                  "payment_mode": ["upi", "netbanking", "auto_debit", "cash"]}
            },
        }
    },

    {
        "name": "investment_v1",
        "domain": "fintech",
        "description": "Retail investment platform with portfolios, trades, and instruments",
        "tables": {
            "investors": {
                "columns": ["investor_id INT PK", "full_name VARCHAR", "pan VARCHAR",
                            "risk_profile VARCHAR", "demat_account VARCHAR",
                            "kyc_date DATE", "is_nri BOOL"],
                "sample_values": {"risk_profile": ["conservative", "moderate", "aggressive"]}
            },
            "instruments": {
                "columns": ["instrument_id INT PK", "symbol VARCHAR", "name VARCHAR",
                            "instrument_type VARCHAR", "exchange VARCHAR",
                            "sector VARCHAR", "market_cap_cr DECIMAL(16,2)"],
                "sample_values": {"instrument_type": ["equity", "mutual_fund", "etf", "bond", "fd"]}
            },
            "portfolios": {
                "columns": ["portfolio_id INT PK", "investor_id INT FK(investors)",
                            "name VARCHAR", "created_at DATE"],
            },
            "holdings": {
                "columns": ["holding_id INT PK", "portfolio_id INT FK(portfolios)",
                            "instrument_id INT FK(instruments)", "quantity DECIMAL(12,4)",
                            "avg_buy_price DECIMAL(12,4)", "current_price DECIMAL(12,4)"],
            },
            "trades": {
                "columns": ["trade_id INT PK", "portfolio_id INT FK(portfolios)",
                            "instrument_id INT FK(instruments)", "trade_type VARCHAR",
                            "quantity DECIMAL(12,4)", "price DECIMAL(12,4)",
                            "brokerage DECIMAL(8,2)", "traded_at TIMESTAMP", "status VARCHAR"],
                "sample_values": {"trade_type": ["buy", "sell"],
                                  "status": ["executed", "cancelled", "pending"]}
            },
        }
    },

    {
        "name": "payments_v1",
        "domain": "fintech",
        "description": "Payment gateway with merchants, settlements, and disputes",
        "tables": {
            "merchants": {
                "columns": ["merchant_id INT PK", "business_name VARCHAR",
                            "category VARCHAR", "mcc_code VARCHAR",
                            "onboarded_date DATE", "risk_level VARCHAR", "active BOOL"],
                "sample_values": {"risk_level": ["low", "medium", "high", "critical"]}
            },
            "payment_links": {
                "columns": ["link_id INT PK", "merchant_id INT FK(merchants)",
                            "amount DECIMAL(12,2)", "currency VARCHAR",
                            "created_at TIMESTAMP", "expires_at TIMESTAMP",
                            "status VARCHAR"],
            },
            "payments": {
                "columns": ["payment_id INT PK", "link_id INT FK(payment_links)",
                            "payer_email VARCHAR", "amount DECIMAL(12,2)",
                            "payment_method VARCHAR", "status VARCHAR",
                            "gateway_txn_id VARCHAR", "paid_at TIMESTAMP"],
                "sample_values": {"payment_method": ["upi", "card", "netbanking", "wallet"],
                                  "status": ["success", "failed", "pending", "refunded"]}
            },
            "settlements": {
                "columns": ["settlement_id INT PK", "merchant_id INT FK(merchants)",
                            "settlement_date DATE", "gross_amount DECIMAL(12,2)",
                            "fees DECIMAL(10,2)", "net_amount DECIMAL(12,2)",
                            "status VARCHAR", "utr_number VARCHAR"],
                "sample_values": {"status": ["processed", "pending", "failed", "held"]}
            },
            "disputes": {
                "columns": ["dispute_id INT PK", "payment_id INT FK(payments)",
                            "raised_by VARCHAR", "reason VARCHAR",
                            "amount DECIMAL(12,2)", "status VARCHAR",
                            "raised_at TIMESTAMP", "resolved_at TIMESTAMP"],
                "sample_values": {"status": ["open", "under_review", "resolved_merchant",
                                             "resolved_customer", "escalated"]}
            },
        }
    },

    {
        "name": "insurance_v1",
        "domain": "fintech",
        "description": "Digital insurance with policies, claims, and agents",
        "tables": {
            "policyholders": {
                "columns": ["holder_id INT PK", "full_name VARCHAR", "dob DATE",
                            "email VARCHAR", "city VARCHAR", "occupation VARCHAR"],
            },
            "insurance_products": {
                "columns": ["product_id INT PK", "name VARCHAR", "type VARCHAR",
                            "min_premium DECIMAL(10,2)", "max_sum_insured DECIMAL(14,2)",
                            "tenure_years INT"],
                "sample_values": {"type": ["term_life", "health", "motor", "travel", "home"]}
            },
            "policies": {
                "columns": ["policy_id INT PK", "holder_id INT FK(policyholders)",
                            "product_id INT FK(insurance_products)", "sum_insured DECIMAL(14,2)",
                            "premium_annual DECIMAL(10,2)", "start_date DATE",
                            "end_date DATE", "status VARCHAR"],
                "sample_values": {"status": ["active", "lapsed", "cancelled", "expired", "claimed"]}
            },
            "claims": {
                "columns": ["claim_id INT PK", "policy_id INT FK(policies)",
                            "claim_type VARCHAR", "claim_amount DECIMAL(12,2)",
                            "filed_date DATE", "status VARCHAR",
                            "settled_amount DECIMAL(12,2)", "settled_date DATE"],
                "sample_values": {"status": ["filed", "under_review", "approved", "rejected", "paid"]}
            },
            "agents": {
                "columns": ["agent_id INT PK", "name VARCHAR", "license_no VARCHAR",
                            "region VARCHAR", "active_policies INT",
                            "commission_pct DECIMAL(4,2)", "joined_date DATE"],
            },
        }
    },

]

ALL_SCHEMAS = ECOMMERCE_SCHEMAS + FINTECH_SCHEMAS


def schema_to_create_sql(schema: dict) -> str:
    """Generate CREATE TABLE statements for a schema (used in prompts)."""
    lines = [f"-- Schema: {schema['name']} ({schema['description']})\n"]
    for table_name, table in schema["tables"].items():
        lines.append(f"CREATE TABLE {table_name} (")
        col_lines = []
        for col in table["columns"]:
            col_lines.append(f"  {col}")
        lines.append(",\n".join(col_lines))
        lines.append(");\n")
    return "\n".join(lines)


def save_schemas():
    index = []
    for schema in ALL_SCHEMAS:
        path = SCHEMAS / f"{schema['name']}.json"
        payload = {**schema, "create_sql": schema_to_create_sql(schema)}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        index.append({
            "name":        schema["name"],
            "domain":      schema["domain"],
            "description": schema["description"],
            "table_count": len(schema["tables"]),
            "path":        str(path),
        })
        print(f"  Saved: {schema['name']} ({schema['domain']}, {len(schema['tables'])} tables)")

    with open(SCHEMAS / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nTotal schemas: {len(ALL_SCHEMAS)} ({sum(1 for s in ALL_SCHEMAS if s['domain']=='ecommerce')} ecommerce, {sum(1 for s in ALL_SCHEMAS if s['domain']=='fintech')} fintech)")
    print(f"Saved → {SCHEMAS}/")


if __name__ == "__main__":
    save_schemas()
    print("\nDay 1 part 2 complete.")
    print("Next: scripts/generate_synthetic.py (Day 3) uses these schemas + GPT-4o to generate 2000+ training pairs.")