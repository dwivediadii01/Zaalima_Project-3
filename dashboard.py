

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import math
from typing import List

st.set_page_config(page_title="Sentiment Dashboard v5", layout="wide")


# Helper functions

@st.cache_data
def load_data(path: str = "sentiment_results.csv"):
    """Load CSV and create safe fallbacks for required columns."""
    df = pd.read_csv(path, low_memory=False)

    # cleaned_text detection
    if "cleaned_text" not in df.columns:
        for alt in ["text", "review", "review_text", "comments", "message", "description"]:
            if alt in df.columns:
                df["cleaned_text"] = df[alt].astype(str)
                break
        else:
            obj_cols = [c for c in df.columns if df[c].dtype == object]
            df["cleaned_text"] = df[obj_cols[0]].astype(str) if obj_cols else df[df.columns[0]].astype(str)

    # label_vader detection
    if "label_vader" not in df.columns:
        for alt in ["label", "sentiment", "label_textblob"]:
            if alt in df.columns:
                df["label_vader"] = df[alt].astype(str)
                break
        else:
            df["label_vader"] = "neutral"

    # numeric score columns safe conversion
    if "vader_compound" in df.columns:
        df["vader_compound"] = pd.to_numeric(df["vader_compound"], errors="coerce")
    if "polarity_textblob" in df.columns:
        df["polarity_textblob"] = pd.to_numeric(df["polarity_textblob"], errors="coerce")

    # parse date-like column if exists
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    if date_cols:
        df["_parsed_date"] = pd.to_datetime(df[date_cols[0]], errors="coerce")
    else:
        df["_parsed_date"] = pd.NaT

    # normalize and ensure types
    df["label_vader"] = df["label_vader"].astype(str).str.lower().replace({"pos": "positive", "neg": "negative"})
    df["cleaned_text"] = df["cleaned_text"].astype(str)
    return df

def make_wordcloud_figure(text: str):
    """Return matplotlib figure for a given text."""
    fig, ax = plt.subplots(figsize=(10,4))
    if not isinstance(text, str) or text.strip() == "":
        ax.text(0.5, 0.5, "No text", ha="center", va="center")
        ax.axis("off")
        return fig
    wc = WordCloud(width=900, height=400, collocations=False).generate(text)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

def top_word_freq(texts: List[str], top_n: int = 20, min_len: int = 3):
    freq = {}
    for t in texts:
        if not isinstance(t, str):
            continue
        for w in t.split():
            if len(w) < min_len:
                continue
            freq[w] = freq.get(w, 0) + 1
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return pd.DataFrame(items, columns=["word", "count"])

def paginate_df(df: pd.DataFrame, page: int, page_size: int = 25):
    start = page * page_size
    end = start + page_size
    return df.iloc[start:end]


# Load data

DATA_PATH = "sentiment_results.csv"
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"File not found: {DATA_PATH}. Place sentiment_results.csv in the app folder.")
    st.stop()


# Sidebar - interactive controls

st.sidebar.header("Interactive Filters (slicers)")

all_labels = sorted(df["label_vader"].dropna().unique().tolist())

# "Select all" convenience checkbox + multiselect
select_all_chk = st.sidebar.checkbox("Select all sentiments", value=True)
if select_all_chk:
    selected_labels = all_labels.copy()
    # still show multiselect but disabled; we set default to all for visibility
    _ = st.sidebar.multiselect("Sentiment (multi-select)", options=all_labels, default=all_labels, disabled=True)
else:
    selected_labels = st.sidebar.multiselect("Sentiment (multi-select)", options=all_labels, default=all_labels)

# Model selector (optional) and score slider (keeps advanced filtering)
available_models = []
if "vader_compound" in df.columns:
    available_models.append("vader")
if "polarity_textblob" in df.columns:
    available_models.append("textblob")
model_choice = st.sidebar.selectbox("Score model (advanced filter)", options=["none"] + available_models, index=0)

# score filter (range) only shown when model selected
if model_choice == "vader":
    min_score = float(np.nanmin(df["vader_compound"].dropna())) if not df["vader_compound"].dropna().empty else -1.0
    max_score = float(np.nanmax(df["vader_compound"].dropna())) if not df["vader_compound"].dropna().empty else 1.0
    score_range = st.sidebar.slider("VADER score range", min_value=-1.0, max_value=1.0, value=(min_score, max_score), step=0.01)
elif model_choice == "textblob":
    min_score = float(np.nanmin(df["polarity_textblob"].dropna())) if not df["polarity_textblob"].dropna().empty else -1.0
    max_score = float(np.nanmax(df["polarity_textblob"].dropna())) if not df["polarity_textblob"].dropna().empty else 1.0
    score_range = st.sidebar.slider("TextBlob polarity range", min_value=-1.0, max_value=1.0, value=(min_score, max_score), step=0.01)
else:
    score_range = None

keyword = st.sidebar.text_input("Keyword filter (contains)")
top_n = st.sidebar.slider("Top words to compute", 5, 50, 15)
possible_drill_cols = [c for c in df.columns if c not in ("cleaned_text","label_vader","vader_compound","polarity_textblob","_parsed_date")]
drill_by = st.sidebar.selectbox("Drill-down / Group by column (optional)", options=["None"] + possible_drill_cols, index=0)

# Date range filter if available
if df["_parsed_date"].notna().any():
    min_d = df["_parsed_date"].min().date()
    max_d = df["_parsed_date"].max().date()
    date_values = st.sidebar.date_input("Date range", value=(min_d, max_d))
    if len(date_values) == 2:
        start_date, end_date = pd.to_datetime(date_values[0]), pd.to_datetime(date_values[1])
    else:
        start_date = end_date = None
else:
    start_date = end_date = None

sample_frac = st.sidebar.slider("Sample fraction for heavy datasets", 0.01, 1.0, 1.0, step=0.01)

st.sidebar.markdown("---")
st.sidebar.write("Tips: use 'Select all' for quick reset. Use model filters for strength-based slicing.")


# Apply filters to dataframe

working = df.copy()

# sentiment filter (if empty or select_all checked -> all)
if selected_labels:
    working = working[working["label_vader"].isin(selected_labels)]

# model score filter
if model_choice == "vader" and score_range is not None:
    working = working[(working["vader_compound"] >= score_range[0]) & (working["vader_compound"] <= score_range[1])]
if model_choice == "textblob" and score_range is not None:
    working = working[(working["polarity_textblob"] >= score_range[0]) & (working["polarity_textblob"] <= score_range[1])]

# keyword filter
if keyword and keyword.strip():
    working = working[working["cleaned_text"].str.contains(keyword.strip(), case=False, na=False)]

# date filter
if start_date is not None and end_date is not None:
    working = working[(working["_parsed_date"] >= start_date) & (working["_parsed_date"] <= end_date)]

# sample
if sample_frac < 1.0:
    working = working.sample(frac=sample_frac, random_state=42)


# KPIs — replaced Avg score with Positive% and Avg review length and Top word

total_reviews = len(df)
filtered_reviews = len(working)
pct = 0 if total_reviews == 0 else filtered_reviews / total_reviews * 100

# Positive percentage in filtered set
if filtered_reviews == 0:
    positive_pct = 0.0
else:
    positive_count = working["label_vader"].astype(str).str.lower().eq("positive").sum()
    positive_pct = positive_count / filtered_reviews * 100

# Avg review length (characters) in filtered
working["review_length"] = working["cleaned_text"].astype(str).apply(len)
avg_length = working["review_length"].mean() if not working["review_length"].dropna().empty else np.nan

# Avg rating (if exists)
rating_cols = [c for c in df.columns if c.lower() in ("rating","stars","score")]
if rating_cols:
    avg_rating = pd.to_numeric(working[rating_cols[0]], errors="coerce").dropna().mean()
else:
    avg_rating = np.nan

# Top word (most frequent token)
texts = working["cleaned_text"].dropna().astype(str).tolist()
top_df = top_word_freq(texts, top_n=1, min_len=3)
top_word = top_df.iloc[0]["word"] if not top_df.empty else "N/A"

col1, col2, col3, col4 = st.columns([1.5,1.5,1.5,1.5])
col1.metric("Total reviews (dataset)", f"{total_reviews:,}")
col2.metric("Filtered reviews", f"{filtered_reviews:,}", f"{pct:.1f}%")
col3.metric("Positive %", f"{positive_pct:.1f}%")
col4.metric("Avg review length", f"{avg_length:.0f} chars" if not np.isnan(avg_length) else "N/A")

# show top word and avg rating below KPIs
with st.container():
    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown(f"**Top word (filtered)**: `{top_word}`")
    with c2:
        if rating_cols:
            st.markdown(f"**Avg {rating_cols[0]} (filtered)**: {avg_rating:.2f}" if not np.isnan(avg_rating) else f"**Avg {rating_cols[0]} (filtered)**: N/A")
        else:
            st.markdown("**Avg rating**: N/A")

st.markdown("---")


# Distribution charts

dist = working["label_vader"].value_counts().reset_index()
dist.columns = ["label", "count"]

left_col, right_col = st.columns([1,1])
with left_col:
    st.subheader("Sentiment share (donut)")
    fig_pie = px.pie(dist, names="label", values="count", hole=0.45, color="label",
                    color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)
with right_col:
    st.subheader("Counts by sentiment")
    fig_bar = px.bar(dist, x="label", y="count", title="Counts by Sentiment", text="count", color="label")
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")


# Drill-down aggregation (if selected)

if drill_by != "None":
    st.subheader(f"Drill-down: counts by `{drill_by}`")
    if working[drill_by].dtype == object or working[drill_by].nunique() > 20:
        agg = working[drill_by].fillna("N/A").value_counts().reset_index()
        agg.columns = [drill_by, "count"]
        agg = agg.head(30)
    else:
        agg = working.groupby(drill_by).size().reset_index(name="count")
    fig_drill = px.bar(agg, x=drill_by, y="count", text="count", title=f"Counts by {drill_by}")
    fig_drill.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_drill, use_container_width=True)
    st.markdown("---")


# Trend (if date available)

if working["_parsed_date"].notna().any():
    st.subheader("Trend over time (by sentiment)")
    tmp = working.dropna(subset=["_parsed_date"]).copy()
    tmp["_date"] = pd.to_datetime(tmp["_parsed_date"]).dt.date
    trend = tmp.groupby(["_date", "label_vader"]).size().reset_index(name="count")
    fig_trend = px.line(trend, x="_date", y="count", color="label_vader", markers=True, title="Daily counts by sentiment")
    st.plotly_chart(fig_trend, use_container_width=True)
    st.markdown("---")


# Top words + interactive selection

st.subheader("Top words (select a word to filter reviews)")

# compute top words from working set
top_words_df = top_word_freq(texts, top_n=top_n, min_len=3)

if not top_words_df.empty:
    fig_words = px.bar(top_words_df, x="word", y="count", text="count", title="Top words")
    fig_words.update_layout(xaxis_tickangle=-45, showlegend=False)
    st.plotly_chart(fig_words, use_container_width=True)
    selected_word = st.selectbox("Pick a word to filter reviews", options=["(none)"] + top_words_df["word"].tolist())
else:
    st.info("No words to display.")
    selected_word = "(none)"

if selected_word and selected_word != "(none)":
    working = working[working["cleaned_text"].str.contains(rf"\b{selected_word}\b", case=False, na=False)]
    st.markdown(f"Filtered reviews containing word **{selected_word}**: {len(working):,}")

wc_text = " ".join(working["cleaned_text"].dropna().astype(str).tolist())
wc_fig = make_wordcloud_figure(wc_text)
st.pyplot(wc_fig)

st.markdown("---")


# Review explorer with pagination & detail viewer

st.subheader("Review Explorer (searchable & paginated)")

search_q = st.text_input("Search in reviews (regex supported)", value="")
if search_q.strip():
    try:
        filtered_table = working[working["cleaned_text"].str.contains(search_q, case=False, na=False, regex=True)]
    except Exception:
        filtered_table = working[working["cleaned_text"].str.contains(search_q, case=False, na=False, regex=False)]
else:
    filtered_table = working

st.write(f"Results: {len(filtered_table):,} rows. Use page controls to browse.")

page_size = st.selectbox("Rows per page", options=[10,20,50,100], index=1)
total_pages = max(1, math.ceil(len(filtered_table) / page_size))
page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
page_idx = page - 1
paginated = paginate_df(filtered_table.reset_index(drop=True), page_idx, page_size=page_size)

display_cols = ["cleaned_text"]
if "vader_compound" in paginated.columns: display_cols.append("vader_compound")
if "polarity_textblob" in paginated.columns: display_cols.append("polarity_textblob")
display_cols.append("label_vader")

st.dataframe(paginated[display_cols], use_container_width=True)

csv_bytes = paginated.to_csv(index=False).encode("utf-8")
st.download_button("Download current page as CSV", csv_bytes, "page_results.csv", "text/csv")

st.markdown("**Open a review detail**")
if not paginated.empty:
    idx_choice = st.selectbox("Select row index on current page", options=list(paginated.index))
    chosen = paginated.loc[idx_choice]
    st.markdown("**Full cleaned text:**")
    st.write(chosen.get("cleaned_text", ""))
    st.markdown("**Other fields:**")
    other_fields = chosen.to_dict()
    for k, v in other_fields.items():
        if k == "cleaned_text": continue
        st.write(f"- **{k}**: {v}")

st.markdown("---")


# Export filtered dataframe (full filtered set)

st.subheader("Export & Summary")
st.write("Download the entire filtered dataset (all rows matching current filters).")
all_csv = filtered_table.to_csv(index=False).encode("utf-8")
st.download_button("Download all filtered results (CSV)", all_csv, "filtered_all_results.csv", "text/csv")

st.markdown("Quick summary:")
summary_df = filtered_table.describe(include="all").transpose()
st.dataframe(summary_df, use_container_width=True)

st.markdown("""
**Notes:**  
- Use 'Select all sentiments' to quickly include all labels.  
- KPIs now show Positive % and Avg review length (these are more informative than a single average model score).  
- If you'd like another KPI (e.g., Net Promoter Score from rating column), tell me which and I will add it.
""")
