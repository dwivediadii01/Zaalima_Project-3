# **Amazon Reviews Sentiment Analysis & Dashboard**

## **Overview**

This project performs end-to-end sentiment analysis on Amazon product reviews and provides an interactive Streamlit dashboard for visual exploration. It includes data preprocessing, sentiment scoring using TextBlob and VADER, and a rich dashboard for KPIs, charts, word clouds, and review-level exploration.

---

## **Project Structure**

<pre class="overflow-visible!" data-start="486" data-end="709"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>.</span><span>
├── </span><span>Amazon</span><span></span><span>Sales</span><span></span><span>Dataset</span><span>.</span><span>csv</span><span>
├── </span><span>preprocess</span><span>.</span><span>py</span><span>
├── </span><span>cleaned_data</span><span>.</span><span>csv</span><span>
├── </span><span>sentiment</span><span>.</span><span>py</span><span>
├── </span><span>sentiment_results</span><span>.</span><span>csv</span><span>
├── </span><span>dashboard</span><span>.</span><span>py</span><span>
├── </span><span>visualizations</span><span>.</span><span>ipynb</span><span>
├── </span><span>Sentiment_Analysis</span><span>_Final</span><span>_Report</span><span>.</span><span>pdf</span><span>
└── </span><span>requirements</span><span>.</span><span>txt</span><span>
</span></span></code></div></div></pre>

---

## **Pipeline**

### **1. Preprocessing (`preprocess.py`)**

* Automatically detects the text column.
* Cleans text by lowercasing, removing URLs/symbols, removing stopwords, and applying lemmatization.
* Produces a new `cleaned_text` column.
* Outputs `cleaned_data.csv`.

### **2. Sentiment Scoring (`sentiment.py`)**

* Generates:
  * `polarity_textblob` (TextBlob polarity)
  * `vader_compound` (VADER compound score)
* Classifies sentiment into:
  * `positive`
  * `neutral`
  * `negative`
* Outputs `sentiment_results.csv`.

### **3. Interactive Dashboard (`dashboard.py`)**

Features include:

* Sentiment slicers with “Select All”
* Keyword, date, drill-down filters
* KPIs:
  * Total reviews
  * Filtered reviews
  * Percent positive
  * Average review length
  * Top word
* Visualizations:
  * Donut chart
  * Bar chart
  * Trend over time
  * Top-word frequency chart
  * Word cloud
* Review Explorer:
  * Regex search
  * Pagination
  * Detailed review viewer
  * CSV export
* Full filtered dataset export

---

## **Dataset Summary**

Based on the uploaded report:

* Total reviews: **1,465**
* Sentiment distribution:
  * Neutral: 1,141
  * Positive: 252
  * Negative: 72
* Word cloud reveals themes like quality, delivery, usability.

---

## **Installation**

### **1. Install dependencies**

<pre class="overflow-visible!" data-start="2051" data-end="2090"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>pip</span><span> install -r requirements.txt
</span></span></code></div></div></pre>

### **2. Run preprocessing**

<pre class="overflow-visible!" data-start="2121" data-end="2149"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>python</span><span> preprocess.py
</span></span></code></div></div></pre>

### **3. Generate sentiment scores**

<pre class="overflow-visible!" data-start="2188" data-end="2215"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>python</span><span> sentiment.py
</span></span></code></div></div></pre>

### **4. Launch the dashboard**

<pre class="overflow-visible!" data-start="2249" data-end="2283"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>streamlit run dashboard.py
</span></span></code></div></div></pre>

---

## **Outputs**

* `cleaned_data.csv` – processed text
* `sentiment_results.csv` – final sentiment-scored dataset
* Interactive Streamlit dashboard
* PDF report summarizing insights

---

## **Recommendations**

* Monitor negative reviews to identify product issues.
* Use positive keywords for marketing messaging.
* Encourage customers to leave more detailed feedback.
* Track sentiment trends over time to detect satisfaction shifts.

---

## **Future Enhancements**

* Aspect-based sentiment analysis
* Automatic review summarization
* Multi-language support
* Trend anomaly detection
