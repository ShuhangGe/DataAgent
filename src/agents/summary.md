# 📋 **Complete Agent Workflow Summary**

## 🔄 **System Overview**

The Sekai Data Analysis Multi-Agent System is a **dynamic, question-answering pipeline** that processes natural language queries and delivers comprehensive analytical insights. The system operates through **6 specialized agents** working in orchestrated sequence, each equipped with specific tools and techniques.

## 🎯 **Agent Pipeline Flow**

```
User Question → Manager → Data Pulling → Preprocessing → Analysis → QA → Insight → Answer
```

---

## 🧠 **1. Manager Agent (Orchestrator)**

**Role:** Question understanding and workflow orchestration

### **Tools & Techniques:**

#### **🔍 QuestionUnderstandingTool**
- **Technique:** **Keyword pattern matching** + **Rule-based classification**
- **Method:** Scans input text for predefined patterns to detect question types
- **Classifications:** 
  - Data exploration: "what data", "show me data", "tables"
  - Statistical summary: "summary", "average", "count", "statistics"
  - Trend analysis: "trend", "over time", "monthly", "growth"
  - Comparison: "compare", "vs", "versus", "between"
  - Correlation: "correlate", "relationship", "factors", "influence"
  - Prediction: "predict", "forecast", "churn", "probability"
- **Entity Extraction:** Maps keywords to database entities (users, events, sessions, revenue)
- **Time Filter Detection:** Regex pattern matching for dates and relative time expressions
- **Confidence Scoring:** Mathematical scoring based on entity count and question type clarity

#### **🗃️ DatabaseSchemaInspectionTool**  
- **Technique:** **SQL metadata inspection** + **Dynamic schema discovery**
- **Method:** Uses SQLAlchemy to introspect database structure
- **Capabilities:**
  - Auto-discovers all tables and column schemas
  - Fetches sample data for context understanding
  - Identifies potential metrics based on column names and data types
  - Filters relevant tables based on question entities

#### **📋 DynamicTaskPlanningTool**
- **Technique:** **Rule-based workflow generation** + **Question-type mapping**
- **Method:** Creates custom agent workflows based on question classification
- **Logic:** Maps question types to specific task sequences with appropriate parameters
- **Planning:** Estimates execution time and determines quality checks needed

---

## 📊 **2. Data Pulling Agent (Data Extractor)**

**Role:** Multi-source data extraction and initial validation

### **Tools & Techniques:**

#### **🔌 DataExtractionTool**
- **Technique:** **Multi-protocol data access** + **Query generation**
- **Supported Sources:**
  - PostgreSQL databases (SQL query generation)
  - CSV files (pandas parsing with encoding detection)
  - DuckDB (columnar analytics with sampling)
  - Parquet files (efficient binary format reading)
- **Method:** Dynamically builds SQL queries or file read operations based on filters
- **Features:** Smart sampling, date range filtering, conditional WHERE clauses

#### **✅ SchemaValidationTool**
- **Technique:** **Statistical validation** + **Business rule checking**
- **Validations:**
  - Required column presence detection
  - Data type consistency verification
  - Quality metric calculation (completeness, consistency, accuracy, timeliness)
  - Volume and shape validation

---

## 🔧 **3. Preprocessing Agent (Data Transformer)**

**Role:** Data cleaning and intelligent feature engineering

### **Tools & Techniques:**

#### **🧹 DataCleaningTool**
- **Technique:** **Statistical cleaning** + **Domain-aware standardization**
- **Missing Value Handling:**
  - Strategy-based filling (mean, median, auto-selection)
  - Column dropping for excessive missing data (>50% threshold)
  - Data type-aware imputation
- **Outlier Management:** Z-score based detection with configurable thresholds, capping instead of removal
- **Standardization:** Timezone normalization, string case standardization, special character cleaning

#### **🏗️ FeatureEngineeringTool**
- **Technique:** **Domain-specific feature generation** + **Gaming analytics patterns**
- **Temporal Features:** Date component extraction, business hour classification, time period categorization
- **Gaming-Specific Features:**
  - Session detection using 30-minute timeout thresholds
  - Event categorization (monetization, engagement, utility events)
  - User lifecycle and tenure calculations
- **Behavioral Features:** Event frequency encoding, user aggregations, cross-session patterns
- **Encoding:** One-hot encoding for categorical variables

---

## 📈 **4. Analysis Agent (Analytics Engine)**

**Role:** Multi-type analytical processing and statistical computation

### **Tools & Techniques:**

#### **🎯 Gaming-Specific Tools:**

**RecommendationFunnelTool:**
- **Technique:** **Event sequence analysis** + **Funnel metrics calculation**
- **Method:** Analyzes user journey from recommendation exposure to clicks
- **Calculations:** Click-through rates, exposure frequencies, user segmentation

**TimePatternAnalysisTool:**
- **Technique:** **Temporal aggregation** + **Pattern detection**
- **Method:** Groups events by time dimensions (hourly, daily) to identify engagement patterns

**UserBehaviorAnalysisTool:**
- **Technique:** **Cohort analysis** + **Behavioral segmentation**
- **Method:** Segments users based on activity levels and analyzes event sequences

#### **🔬 Dynamic Analysis Tools:**

**StatisticalSummaryTool:**
- **Technique:** **Descriptive statistics** + **Data profiling**
- **Method:** Calculates comprehensive statistical summaries using pandas aggregations

**TrendAnalysisTool:**
- **Technique:** **Time series analysis** + **Trend detection algorithms**
- **Method:** Analyzes temporal patterns and calculates trend metrics

---

## ✅ **5. QA Agent (Quality Gatekeeper)**

**Role:** Comprehensive validation and quality assurance

### **Tools & Techniques:**

#### **🔍 DataQualityValidationTool**
- **Technique:** **Multi-level validation framework** + **Statistical quality assessment**
- **Basic Validations:**
  - Empty dataset detection
  - Required column verification
  - Duplicate row identification
  - Data type consistency checking
- **Statistical Validations:**
  - Z-score based outlier detection
  - Timestamp distribution analysis
  - Data gap identification
- **Business Rule Engine:**
  - Value range constraints
  - Allowed values validation
  - Pattern matching (regex-based)
  - Cross-field relationship validation

#### **📋 ResultValidationTool**
- **Technique:** **Result integrity verification** + **Business logic validation**
- **Completeness Checks:** Ensures all required analysis components are present
- **Accuracy Validation:** 
  - Mathematical consistency verification
  - Range validation for calculated metrics
  - Logical relationship verification (e.g., retention rates should decrease over time)
- **Business Logic:** Industry-specific validation rules for gaming metrics

---

## 💡 **6. Insight Agent (Intelligence Synthesizer)**

**Role:** Business insight generation and recommendation synthesis

### **Tools & Techniques:**

#### **🧠 RecommendationInsightTool**
- **Technique:** **Rule-based insight generation** + **Business intelligence patterns**
- **Insight Categories:**
  - Performance assessment using industry benchmarks
  - Optimization opportunity identification
  - Risk factor analysis
  - Data quality issue flagging
- **Recommendation Engine:**
  - Priority-based action item generation
  - Impact estimation and effort assessment
  - Implementation strategy suggestions
- **Insight Synthesis:**
  - Confidence scoring for insights
  - Executive summary generation
  - Actionable recommendation prioritization

---

## 🎯 **Key Techniques Summary**

### **Natural Language Processing:**
- **Pattern Matching:** Keyword-based question classification
- **Entity Recognition:** Rule-based extraction of database entities
- **Intent Detection:** Question type classification using predefined patterns

### **Data Processing:**
- **SQL Generation:** Dynamic query building based on filters
- **Statistical Cleaning:** Z-score outlier detection, smart imputation
- **Feature Engineering:** Domain-aware temporal and behavioral feature creation

### **Quality Assurance:**
- **Multi-tier Validation:** Basic, statistical, and business rule validation
- **Quality Scoring:** Mathematical quality metric calculation
- **Anomaly Detection:** Statistical and rule-based anomaly identification

### **Business Intelligence:**
- **Benchmark Comparison:** Industry standard comparison for insight generation
- **Rule-based Insights:** Predefined logic for insight classification
- **Priority-based Recommendations:** Automated action item generation with impact assessment

## 🔄 **Dynamic Adaptation**

The system **adapts its processing approach** based on question type:

- **Data Exploration:** Minimal processing, schema focus
- **Statistical Summary:** Full preprocessing, descriptive analytics
- **Trend Analysis:** Enhanced temporal features, time series analysis
- **Comparison:** Group-level aggregations, comparative metrics
- **Correlation:** Advanced feature engineering, relationship analysis
- **Prediction:** Maximum feature richness, predictive modeling

This creates a **truly dynamic, intelligent system** that processes any natural language question about data and delivers comprehensive, actionable insights through sophisticated multi-agent orchestration.
