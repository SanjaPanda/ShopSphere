# ShopSphere AI – Architecture

## High-Level Architecture

```text
                    User
                      │
              Next.js Web App
                      │
                FastAPI Backend
                      │
             LangGraph Orchestrator
                      │
      ┌────────┬─────────┬─────────┐
      │        │         │         │
 Planner    SQL Agent  RAG Agent Validation
      │        │         │         │
      └────────┴─────────┴─────────┘
                      │
              Explanation Agent
                      │
               Final Response
```

---

# Data Layer

## Structured Data

PostgreSQL

Raw tables

* customers
* orders
* order_items
* payments
* reviews
* sellers
* products
* geolocation

Analytics tables (dbt)

* customer_features
* seller_features
* product_features
* logistics_features
* sales_summary
* monthly_metrics

---

# Knowledge Layer

Enterprise documents

* Shipping Policy
* Return Policy
* Seller Handbook
* Customer Support FAQ
* Warehouse SOP
* Quarterly Reports
* Marketing Strategy
* Inventory Guidelines

Pipeline

```text
Documents
     │
Chunking
     │
Embeddings
     │
Vector Database
     │
Retriever
```

---

# AI Agents

## Planner Agent

Responsibilities

* Detect user intent
* Route requests
* Choose SQL, RAG, or Hybrid execution
* Ask clarification questions when needed

---

## SQL Agent

Responsibilities

* Generate SQL
* Validate SQL
* Execute SQL
* Explain SQL

Data Source

PostgreSQL + dbt feature tables

---

## RAG Agent

Responsibilities

* Retrieve relevant documents
* Rank results
* Provide citations
* Supply business context

Data Source

Vector Database

---

## Validation Agent

Responsibilities

* Validate SQL results
* Validate retrieved evidence
* Detect conflicting information
* Produce confidence score

---

## Explanation Agent

Responsibilities

* Merge SQL and RAG outputs
* Produce business-friendly explanations
* Generate final response

---

# Backend Components

* FastAPI
* SQLAlchemy
* LangGraph
* Prompt templates
* Retrieval service
* SQL service
* Authentication
* Logging

---

# Observability

Capture

* User request
* Agent execution
* SQL queries
* Retrieved documents
* Tokens
* Latency
* Cost
* Errors

Tools

* LangSmith
* OpenTelemetry

---

# Evaluation

Metrics

* SQL accuracy
* Retrieval precision
* Hallucination rate
* Latency
* Cost per request
* User satisfaction

---

# User Interface

Features

* Chat interface
* Conversation history
* SQL viewer
* Retrieved document viewer
* Agent execution trace
* Token usage
* Cost dashboard
* Feedback collection

---

# Future Enhancements

* Role-based access control
* Dashboard generation
* Chart generation
* Scheduled reports
* Multi-modal document support
* Voice interface
* Multi-language support
* Agent marketplace

---

# Design Principles

1. Modular architecture
2. Production-ready components
3. Observable by default
4. Easy to test
5. Easy to extend
6. Cloud-ready deployment
7. Clear separation between data, AI, and presentation layers
