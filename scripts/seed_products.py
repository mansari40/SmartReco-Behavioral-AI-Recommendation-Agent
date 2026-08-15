"""
Seeds the product catalog by calling the real API (not writing to the DB
directly) — this exercises the actual dual-write path (SQL + embed +
Chroma) exactly the way the admin console does, just faster and repeatable.

Usage: python -m scripts.seed_products admin@smartreco.com admin1234
"""
import json
import sys
import urllib.error
import urllib.request

import os
BASE_URL = os.environ.get("SEED_TARGET_URL", "http://localhost:8000")


CATALOG = [
    {
        "title": "Agentic AI Fundamentals",
        "description": "Learn to build reasoning agents with LangGraph and RAG pipelines. Covers state machines, tool use, retrieval-augmented generation, and multi-step reasoning workflows.",
        "category": "AI",
        "price": 49.99,
    },
    {
        "title": "Production RAG at Scale",
        "description": "Advanced retrieval-augmented generation patterns for real-world systems — hybrid search, re-ranking, metadata filtering, chunking strategies, and evaluation.",
        "category": "AI",
        "price": 79.99,
    },
    {
        "title": "MLOps with Kubernetes",
        "description": "Deploy and scale machine learning systems in production using Kubernetes, covering CI/CD for models, container orchestration, monitoring, versioning, and rollback strategies.",
        "category": "AI",
        "price": 89.99,
    },
    {
        "title": "Prompt Engineering for Enterprises",
        "description": "Structured prompting techniques for reliable, testable LLM outputs at scale — few-shot design, structured outputs, evaluation harnesses, and guardrails.",
        "category": "AI",
        "price": 59.99,
    },
    {
        "title": "Building LLM Applications with Python",
        "description": "Build practical large language model applications with Python, APIs, structured outputs, function calling, streaming, and production-ready application patterns.",
        "category": "AI",
        "price": 64.99,
    },
    {
        "title": "Fine-Tuning Language Models",
        "description": "Learn how to adapt pretrained language models for specialized tasks using supervised fine-tuning, datasets, parameter-efficient methods, evaluation, and deployment.",
        "category": "AI",
        "price": 84.99,
    },
    {
        "title": "AI Agents with Tool Calling",
        "description": "Design AI agents that interact with external tools and APIs. Covers tool schemas, planning, execution loops, error handling, and reliable agent workflows.",
        "category": "AI",
        "price": 69.99,
    },
    {
        "title": "Multimodal AI Applications",
        "description": "Build applications that understand text, images, documents, and other modalities using modern multimodal AI models and practical application architectures.",
        "category": "AI",
        "price": 74.99,
    },
    {
        "title": "Data Engineering with Airflow & Spark",
        "description": "Build resilient batch data pipelines with Airflow orchestration and Spark processing — handling schema drift, backfills, dependencies, monitoring, and cost control.",
        "category": "Data Engineering",
        "price": 69.99,
    },
    {
        "title": "Modern Data Pipelines with Python",
        "description": "Learn how to design reliable ETL and ELT pipelines using Python, APIs, databases, validation, logging, scheduling, and automated data quality checks.",
        "category": "Data Engineering",
        "price": 54.99,
    },
    {
        "title": "Apache Spark for Data Processing",
        "description": "Process large datasets efficiently with Apache Spark. Covers DataFrames, Spark SQL, distributed transformations, joins, optimization, and partitioning.",
        "category": "Data Engineering",
        "price": 64.99,
    },
    {
        "title": "Data Warehousing Fundamentals",
        "description": "Understand dimensional modeling, fact and dimension tables, star schemas, slowly changing dimensions, ETL pipelines, and analytical data warehouses.",
        "category": "Data Engineering",
        "price": 44.99,
    },
    {
        "title": "dbt Analytics Engineering",
        "description": "Transform raw warehouse data into reliable analytical models with dbt. Covers SQL transformations, testing, documentation, dependencies, and deployment workflows.",
        "category": "Data Engineering",
        "price": 59.99,
    },
    {
        "title": "Streaming Data with Kafka",
        "description": "Build real-time data pipelines using Apache Kafka. Learn topics, partitions, consumers, producers, event-driven architectures, delivery guarantees, and monitoring.",
        "category": "Data Engineering",
        "price": 74.99,
    },
    {
        "title": "Data Quality Engineering",
        "description": "Build trustworthy data systems using validation rules, anomaly detection, schema checks, freshness monitoring, lineage, and automated quality testing.",
        "category": "Data Engineering",
        "price": 49.99,
    },
    {
        "title": "Python for Data Analysis",
        "description": "Master practical data analysis with Python, pandas, NumPy, data cleaning, joins, grouping, reshaping, missing values, and exploratory analysis.",
        "category": "Data Science",
        "price": 39.99,
    },
    {
        "title": "Statistics for Data Science",
        "description": "Learn probability, distributions, hypothesis testing, confidence intervals, correlation, regression, and statistical reasoning for data science applications.",
        "category": "Data Science",
        "price": 44.99,
    },
    {
        "title": "Machine Learning with Scikit-Learn",
        "description": "Build supervised and unsupervised machine learning models with scikit-learn, including preprocessing, regression, classification, clustering, and model evaluation.",
        "category": "Data Science",
        "price": 54.99,
    },
    {
        "title": "Deep Learning with PyTorch",
        "description": "Learn neural networks and deep learning with PyTorch. Covers tensors, datasets, training loops, optimization, regularization, and model evaluation.",
        "category": "Data Science",
        "price": 69.99,
    },
    {
        "title": "Practical Natural Language Processing",
        "description": "Process and analyze text using tokenization, embeddings, sentiment analysis, named entity recognition, classification, and modern NLP techniques.",
        "category": "Data Science",
        "price": 59.99,
    },
    {
        "title": "Time Series Forecasting",
        "description": "Learn forecasting techniques for business and operational data, including trend analysis, seasonality, ARIMA, feature engineering, validation, and model comparison.",
        "category": "Data Science",
        "price": 54.99,
    },
    {
        "title": "Feature Engineering Masterclass",
        "description": "Improve machine learning performance through effective feature creation, transformation, encoding, selection, leakage prevention, and domain-driven feature design.",
        "category": "Data Science",
        "price": 49.99,
    },
    {
        "title": "Model Evaluation and Experimentation",
        "description": "Learn rigorous machine learning evaluation using cross-validation, precision, recall, F1, ROC-AUC, calibration, error analysis, and experiment tracking.",
        "category": "Data Science",
        "price": 46.99,
    },
    {
        "title": "SQL for Data Analysts",
        "description": "Master SQL for analytical work using filtering, joins, aggregations, subqueries, window functions, CTEs, and practical business reporting problems.",
        "category": "Data Science",
        "price": 34.99,
    },
    {
        "title": "Advanced SQL Analytics",
        "description": "Go beyond basic SQL with window functions, cohort analysis, retention metrics, ranking, recursive queries, query optimization, and analytical reporting.",
        "category": "Data Science",
        "price": 49.99,
    },
    {
        "title": "Python Programming from Scratch",
        "description": "Learn Python fundamentals including variables, functions, collections, loops, modules, exceptions, file handling, and object-oriented programming.",
        "category": "Programming",
        "price": 29.99,
    },
    {
        "title": "Advanced Python Development",
        "description": "Write professional Python applications using decorators, generators, context managers, typing, asynchronous programming, testing, and clean architecture.",
        "category": "Programming",
        "price": 59.99,
    },
    {
        "title": "FastAPI Backend Development",
        "description": "Build modern Python APIs with FastAPI, including authentication, validation, dependency injection, database integration, asynchronous endpoints, and API documentation.",
        "category": "Programming",
        "price": 54.99,
    },
    {
        "title": "React and TypeScript Fundamentals",
        "description": "Build modern frontend applications with React and TypeScript. Covers components, state, props, hooks, forms, API integration, and reusable UI patterns.",
        "category": "Programming",
        "price": 49.99,
    },
    {
        "title": "Next.js Full-Stack Development",
        "description": "Build production-ready web applications with Next.js and TypeScript using routing, server components, APIs, authentication, data fetching, and deployment.",
        "category": "Programming",
        "price": 69.99,
    },
    {
        "title": "Clean Code and Software Architecture",
        "description": "Learn practical software engineering principles for maintainable applications, including modular design, separation of concerns, dependency management, testing, and refactoring.",
        "category": "Programming",
        "price": 44.99,
    },
    {
        "title": "Docker for Developers",
        "description": "Containerize applications with Docker. Covers images, containers, Dockerfiles, volumes, networks, Compose, environment configuration, and production practices.",
        "category": "Cloud & DevOps",
        "price": 39.99,
    },
    {
        "title": "Kubernetes for Beginners",
        "description": "Understand Kubernetes fundamentals including pods, deployments, services, ingress, configuration, secrets, scaling, health checks, and cluster management.",
        "category": "Cloud & DevOps",
        "price": 59.99,
    },
    {
        "title": "AWS Cloud Practitioner",
        "description": "Learn the foundations of cloud computing with AWS, including compute, storage, networking, databases, security, pricing, and core cloud architecture concepts.",
        "category": "Cloud & DevOps",
        "price": 44.99,
    },
    {
        "title": "Google Cloud Fundamentals",
        "description": "Explore Google Cloud services including Compute Engine, Cloud Run, Cloud Storage, BigQuery, IAM, networking, monitoring, and serverless application deployment.",
        "category": "Cloud & DevOps",
        "price": 44.99,
    },
    {
        "title": "CI/CD with GitHub Actions",
        "description": "Automate software testing and deployment with GitHub Actions. Build workflows for linting, testing, Docker images, releases, secrets, and cloud deployments.",
        "category": "Cloud & DevOps",
        "price": 42.99,
    },
    {
        "title": "Terraform Infrastructure as Code",
        "description": "Manage cloud infrastructure using Terraform. Learn providers, resources, variables, modules, state management, environments, and infrastructure automation.",
        "category": "Cloud & DevOps",
        "price": 54.99,
    },
    {
        "title": "Cybersecurity Fundamentals",
        "description": "Understand core cybersecurity concepts including threats, vulnerabilities, authentication, encryption, network security, access control, and security best practices.",
        "category": "Cybersecurity",
        "price": 39.99,
    },
    {
        "title": "Web Application Security",
        "description": "Learn how modern web applications are protected against common vulnerabilities including injection, broken authentication, access control issues, and insecure configurations.",
        "category": "Cybersecurity",
        "price": 54.99,
    },
    {
        "title": "Ethical Hacking Fundamentals",
        "description": "Learn the principles and methodology of authorized security testing, including reconnaissance, vulnerability assessment, threat modeling, and security reporting.",
        "category": "Cybersecurity",
        "price": 64.99,
    },
    {
        "title": "Cybersecurity Risk Management",
        "description": "Learn how organizations identify, assess, prioritize, and mitigate cybersecurity risks using structured risk frameworks, controls, policies, and incident planning.",
        "category": "Cybersecurity",
        "price": 49.99,
    },
    {
        "title": "Business Analytics Fundamentals",
        "description": "Turn business data into actionable insights using KPIs, descriptive analytics, dashboards, segmentation, trend analysis, and data-driven decision making.",
        "category": "Business",
        "price": 34.99,
    },
    {
        "title": "Financial Modeling with Excel",
        "description": "Build practical financial models in Excel using assumptions, forecasts, income statements, cash flows, scenario analysis, valuation, and sensitivity analysis.",
        "category": "Business",
        "price": 49.99,
    },
    {
        "title": "Product Management Essentials",
        "description": "Learn product discovery, customer research, prioritization, roadmapping, product metrics, experimentation, stakeholder management, and product strategy.",
        "category": "Business",
        "price": 44.99,
    },
    {
        "title": "Digital Marketing Analytics",
        "description": "Measure digital marketing performance using acquisition metrics, conversion funnels, attribution, customer segmentation, campaign analysis, and marketing dashboards.",
        "category": "Business",
        "price": 39.99,
    },
    {
        "title": "Data Visualization with Tableau",
        "description": "Create effective interactive dashboards with Tableau using calculated fields, filters, parameters, charts, storytelling techniques, and business intelligence principles.",
        "category": "Design & Visualization",
        "price": 49.99,
    },
    {
        "title": "Data Visualization with Power BI",
        "description": "Build interactive Power BI dashboards using data modeling, DAX, Power Query, relationships, calculated measures, and effective business reporting techniques.",
        "category": "Design & Visualization",
        "price": 54.99,
    },
    {
        "title": "UI/UX Design Fundamentals",
        "description": "Learn user-centered design principles, wireframing, information architecture, usability testing, interaction design, and practical interface design workflows.",
        "category": "Design & Visualization",
        "price": 39.99,
    },
    {
        "title": "Introduction to Baking",
        "description": "Learn to bake bread and pastries at home — fermentation basics, laminated dough, ingredient ratios, oven techniques, and troubleshooting common failures.",
        "category": "Culinary",
        "price": 19.99,
    },
    {
        "title": "Italian Cooking at Home",
        "description": "Learn classic Italian cooking techniques including fresh pasta, risotto, sauces, roasted vegetables, seafood, and traditional regional dishes.",
        "category": "Culinary",
        "price": 24.99,
    },
    {
        "title": "Healthy Meal Preparation",
        "description": "Learn practical meal-preparation techniques for balanced everyday meals, including planning, batch cooking, nutritious ingredients, storage, and portion management.",
        "category": "Culinary",
        "price": 22.99,
    },
    {
        "title": "Artisan Bread Masterclass",
        "description": "Master naturally fermented artisan bread with sourdough starters, hydration control, gluten development, shaping, proofing, scoring, and baking techniques.",
        "category": "Culinary",
        "price": 29.99,
    },
]




def login(email: str, password: str) -> str:
    body = f"username={email}&password={password}".encode()
    req = urllib.request.Request(
        f"{BASE_URL}/api/auth/login",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)["access_token"]


def create_product(token: str, product: dict) -> None:
    req = urllib.request.Request(
        f"{BASE_URL}/api/products",
        data=json.dumps(product).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.load(resp)
        print(f"  Created: {result['title']} (sync_status: {result['sync_status']})")


def main():
    if len(sys.argv) != 3:
        print("Usage: python -m scripts.seed_products <admin-email> <admin-password>")
        sys.exit(1)

    email, password = sys.argv[1], sys.argv[2]

    print("Logging in...")
    try:
        token = login(email, password)
    except urllib.error.HTTPError as e:
        print(f"Login failed: {e.code} {e.reason}")
        print(e.read().decode())
        sys.exit(1)

    print(f"Seeding {len(CATALOG)} products...")
    for product in CATALOG:
        try:
            create_product(token, product)
        except urllib.error.HTTPError as e:
            print(f"  Failed to create '{product['title']}': {e.code} {e.read().decode()}")

    print("Done.")


if __name__ == "__main__":
    main()