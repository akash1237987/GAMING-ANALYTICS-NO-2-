import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Gaming Industry Analytics",
    page_icon="🎮",
    layout="wide"
)


# ============================================================
# MODULE 1
# DATA COLLECTION, CLEANING, HARMONIZATION
# AND FEATURE ENGINEERING
# ============================================================

def create_gaming_data():

    data = {

        "game_name": [
            "God of War",
            "Minecraft",
            "GTA V",
            "FIFA 23",
            "The Witcher 3",
            "Red Dead Redemption 2",
            "Fortnite",
            "PUBG",
            "Elden Ring",
            "Mario Kart 8",
            "Pokemon Scarlet",
            "Halo Infinite",
            "Forza Horizon 5",
            "Spider-Man",
            "Cyberpunk 2077",
            "Valorant",
            "Assassin's Creed Valhalla",
            "Resident Evil Village",
            "Call of Duty MW2",
            "Hogwarts Legacy"
        ],

        "platform": [
            "PS4",
            "PC",
            "PS4",
            "PS4",
            "PC",
            "PS4",
            "PC",
            "PC",
            "PS5",
            "Switch",
            "Switch",
            "XONE",
            "XONE",
            "PS4",
            "PC",
            "PC",
            "PS5",
            "PS5",
            "PS5",
            "PS5"
        ],

        "genre": [
            "Action",
            "Sandbox",
            "Action",
            "Sports",
            "RPG",
            "Action",
            "Battle Royale",
            "Battle Royale",
            "RPG",
            "Racing",
            "RPG",
            "Shooter",
            "Racing",
            "Action",
            "RPG",
            "Shooter",
            "Action",
            "Horror",
            "Shooter",
            "RPG"
        ],

        "year": [
            2018,
            2011,
            2014,
            2022,
            2015,
            2018,
            2017,
            2017,
            2022,
            2017,
            2022,
            2021,
            2021,
            2018,
            2020,
            2020,
            2020,
            2021,
            2022,
            2023
        ],

        "sales": [
            20,
            30,
            40,
            12,
            15,
            25,
            35,
            28,
            18,
            14,
            10,
            8,
            11,
            16,
            13,
            20,
            12,
            10,
            22,
            15
        ],

        "rating": [
            9.5,
            9.0,
            9.7,
            7.8,
            9.6,
            9.8,
            8.5,
            8.2,
            9.6,
            9.0,
            7.5,
            8.0,
            9.0,
            9.2,
            8.0,
            8.5,
            8.4,
            8.7,
            8.8,
            8.6
        ],

        "critic_score": [
            94,
            93,
            97,
            78,
            93,
            97,
            81,
            86,
            96,
            92,
            72,
            87,
            92,
            87,
            86,
            80,
            84,
            84,
            83,
            84
        ],

        "user_score": [
            9.3,
            9.0,
            9.5,
            7.5,
            9.4,
            9.7,
            8.7,
            8.1,
            9.5,
            9.1,
            7.8,
            7.5,
            8.8,
            9.0,
            7.8,
            8.4,
            8.2,
            8.5,
            8.4,
            8.3
        ]
    }

    df = pd.DataFrame(data)

    return df


# ------------------------------------------------------------
# DATA CLEANING
# ------------------------------------------------------------

def clean_data(df):
    """Normalize and validate uploaded or sample gaming data."""
    df = df.copy()
    df = df.drop_duplicates()

    # Normalize column names, for example: "Game Name" -> "game_name".
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    required_columns = ["game_name", "platform", "genre", "year", "sales", "rating"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "The CSV is missing these required columns: " + ", ".join(missing)
        )

    # These two scores are optional in user-uploaded files.
    for column in ["critic_score", "user_score"]:
        if column not in df.columns:
            df[column] = np.nan

    numeric_columns = ["year", "sales", "rating", "critic_score", "user_score"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["game_name"] = df["game_name"].fillna("Unknown Game").astype(str).str.strip()
    df["genre"] = df["genre"].fillna("Unknown").astype(str).str.strip()
    df["platform"] = df["platform"].fillna("Unknown").astype(str).str.strip()
    df["sales"] = df["sales"].fillna(0)

    average_rating = df["rating"].mean()
    df["rating"] = df["rating"].fillna(average_rating if pd.notna(average_rating) else 0)
    df["year"] = df["year"].fillna(0).astype(int)

    return df


# ------------------------------------------------------------
# PLATFORM HARMONIZATION
# ------------------------------------------------------------

def harmonize_platforms(df):

    platform_mapping = {

        "PS1": "PlayStation",
        "PS2": "PlayStation 2",
        "PS3": "PlayStation 3",
        "PS4": "PlayStation 4",
        "PS5": "PlayStation 5",

        "X360": "Xbox 360",
        "XONE": "Xbox One",
        "XBONE": "Xbox One",

        "PC": "PC",

        "Wii": "Nintendo Wii",
        "WiiU": "Nintendo Wii U",
        "Switch": "Nintendo Switch"
    }

    df["platform"] = df["platform"].replace(
        platform_mapping
    )

    return df


# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------

def feature_engineering(df):

    # Popularity score
    df["popularity_score"] = (
        df["sales"] * 0.6
        +
        df["rating"] * 0.4
    )

    # Rating category
    def get_rating_category(rating):

        if rating >= 8:
            return "Excellent"

        elif rating >= 6:
            return "Good"

        elif rating >= 4:
            return "Average"

        else:
            return "Poor"

    df["rating_category"] = (
        df["rating"]
        .apply(get_rating_category)
    )

    # Sales category
    def get_sales_category(sales):

        if sales >= 20:
            return "Blockbuster"

        elif sales >= 10:
            return "High"

        elif sales >= 1:
            return "Medium"

        else:
            return "Low"

    df["sales_category"] = (
        df["sales"]
        .apply(get_sales_category)
    )

    return df


# ============================================================
# COMPLETE MODULE 1 PIPELINE
# ============================================================

def process_data(data=None):
    """Prepare either the built-in sample or data supplied by the user."""
    if data is None:
        data = create_gaming_data()

    df = clean_data(data)
    df = harmonize_platforms(df)
    df = feature_engineering(df)
    return df


# ============================================================
# MODULE 2
# MULTI-PLATFORM GAMING TRENDS
# ============================================================

def yearly_trends(df):

    result = (
        df.groupby("year")
        .agg(
            total_games=("game_name", "count"),
            total_sales=("sales", "sum"),
            average_rating=("rating", "mean")
        )
        .reset_index()
    )

    return result


def platform_trends(df):

    result = (
        df.groupby("platform")
        .agg(
            total_games=("game_name", "count"),
            total_sales=("sales", "sum"),
            average_rating=("rating", "mean")
        )
        .reset_index()
        .sort_values(
            "total_sales",
            ascending=False
        )
    )

    return result


def popularity_analysis(df):

    result = (
        df.sort_values(
            "popularity_score",
            ascending=False
        )
    )

    return result


def platform_market_share(df):

    result = (
        df.groupby("platform")["sales"]
        .sum()
        .reset_index()
    )

    total_sales = result["sales"].sum()

    result["market_share"] = (
        result["sales"]
        /
        total_sales
        *
        100
    )

    return result


# ============================================================
# MODULE 3
# GENRE, PLATFORM, RATINGS & SALES
# COMPARATIVE ANALYTICS
# ============================================================

def genre_analysis(df):

    result = (
        df.groupby("genre")
        .agg(
            total_games=("game_name", "count"),
            total_sales=("sales", "sum"),
            average_rating=("rating", "mean")
        )
        .reset_index()
        .sort_values(
            "total_sales",
            ascending=False
        )
    )

    return result


def rating_analysis(df):

    result = (
        df.groupby("rating_category")
        .agg(
            games=("game_name", "count"),
            sales=("sales", "sum")
        )
        .reset_index()
    )

    return result


def platform_rating_comparison(df):

    result = (
        df.groupby("platform")
        .agg(
            average_rating=("rating", "mean"),
            total_sales=("sales", "sum"),
            number_of_games=("game_name", "count")
        )
        .reset_index()
    )

    return result


def genre_platform_analysis(df):

    result = pd.pivot_table(

        df,

        values="sales",

        index="genre",

        columns="platform",

        aggfunc="sum",

        fill_value=0
    )

    return result


# ============================================================
# PROJECT DATA (SESSION-BASED, USER-CHANGEABLE)
# ============================================================

if "gaming_df" not in st.session_state:
    st.session_state.gaming_df = process_data()
if "dataset_revision" not in st.session_state:
    st.session_state.dataset_revision = 0

# Refresh this on every Streamlit rerun so all charts use current user data.
df = st.session_state.gaming_df

if "flash_message" in st.session_state:
    st.success(st.session_state.pop("flash_message"))

# ============================================================
# MODULE 2 – SIGN IN / LOGOUT
# ============================================================
DEMO_USERNAME = "student"
DEMO_PASSWORD = "1234"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""


def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()


# ============================================================
# MODULE 1 – HOMEPAGE
# ============================================================
def show_homepage():
    st.title("🎮 Gaming Industry Analytics")
    st.subheader("Multi-Platform Gaming Trends, Popularity, Ratings & Sales Analytics")
    st.write(
        "This project uses Python, Streamlit and Data Analytics to analyze "
        "gaming-industry data and present interactive insights."
    )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.info("""
### 📊 What this project analyzes
- Gaming sales
- Game ratings
- Platforms
- Genres
- Year-wise trends
- Platform market share
- Popularity
- Comparative analytics
""")

    with col2:
        st.success("""
### 🧑‍💻 Project Team
**Akash Bhardwaj**  
**Keshav Kumar Sahni**

**Technology:** Python + Streamlit + Data Analytics
""")

    st.markdown("---")
    st.subheader("📚 Project Structure")
    st.markdown("""
**Module 1 – Homepage**  
Introduction, project information and navigation.

**Module 2 – Sign In / Logout**  
User authentication and logout functionality.

**Module 3 – Main Content**  
Gaming trends, popularity and sales analytics.

**Module 4 – Second Analytics Content**  
Genre, platform, rating and comparative analytics.
""")


# ============================================================
# MODULE 2 – LOGIN PAGE
# ============================================================
def show_login():
    st.title("🔐 Sign In")
    st.write("Sign in to access the gaming analytics dashboard.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In", use_container_width=True)

    if submitted:
        if username == DEMO_USERNAME and password == DEMO_PASSWORD:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful! 🎮")
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.caption("Demo login: username = student | password = 1234")


# ============================================================
# MODULE 3 – MAIN CONTENT
# ============================================================
def show_main_content():
    st.title("📊 Module 3 – Main Gaming Analytics")
    st.write("Gaming trends, popularity, sales and platform-level analytics.")

    st.sidebar.header("🎮 Analytics Filters")
    platforms = sorted(df["platform"].unique())
    selected_platforms = st.sidebar.multiselect(
        "Select Platform", platforms, default=platforms
    )
    genres = sorted(df["genre"].unique())
    selected_genres = st.sidebar.multiselect(
        "Select Genre", genres, default=genres
    )
    minimum_rating = st.sidebar.slider(
        "Minimum Rating", min_value=0.0, max_value=10.0,
        value=0.0, step=0.5
    )

    filtered_df = df[
        (df["platform"].isin(selected_platforms)) &
        (df["genre"].isin(selected_genres)) &
        (df["rating"] >= minimum_rating)
    ]

    st.header("📈 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎮 Total Games", len(filtered_df))
    with col2:
        st.metric("💰 Total Sales", round(filtered_df["sales"].sum(), 2))
    with col3:
        avg = filtered_df["rating"].mean()
        st.metric("⭐ Average Rating", round(avg, 2) if not pd.isna(avg) else 0)
    with col4:
        st.metric("🎯 Platforms", filtered_df["platform"].nunique())

    st.markdown("---")
    st.subheader("📈 Games Released by Year")
    yearly = yearly_trends(filtered_df)
    if not yearly.empty:
        fig1 = px.line(yearly, x="year", y="total_games", markers=True,
                       title="Games Released by Year")
        st.plotly_chart(fig1, use_container_width=True)

    st.subheader("💰 Sales by Platform")
    platform = platform_trends(filtered_df)
    if not platform.empty:
        fig2 = px.bar(platform, x="platform", y="total_sales", text_auto=True,
                      title="Sales by Platform")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🔥 Game Popularity Analytics")
    popularity = popularity_analysis(filtered_df)
    if not popularity.empty:
        fig3 = px.scatter(popularity, x="rating", y="sales",
                          size="popularity_score", hover_name="game_name",
                          title="Game Rating vs Sales")
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("🏆 Top 10 Popular Games")
        st.dataframe(
            popularity[["game_name", "platform", "genre", "sales",
                        "rating", "popularity_score"]].head(10),
            use_container_width=True
        )

        market = platform_market_share(filtered_df)
        if not market.empty:
            st.subheader("🥧 Platform Market Share")
            fig4 = px.pie(market, names="platform", values="market_share",
                          title="Platform Market Share")
            st.plotly_chart(fig4, use_container_width=True)

    st.subheader("📋 Filtered Dataset")
    st.dataframe(filtered_df, use_container_width=True)
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download Processed Data", csv,
        "gaming_analysis_data.csv", "text/csv"
    )


# ============================================================
# MODULE 4 – SECOND ANALYTICS CONTENT
# ============================================================
def show_comparative_content():
    st.title("🎯 Module 4 – Comparative Analytics")
    st.write("Compare genres, platforms, ratings and sales to identify patterns.")

    selected_platforms = st.multiselect(
        "Select Platforms for Comparison", sorted(df["platform"].unique()),
        default=sorted(df["platform"].unique()), key="comparison_platforms"
    )
    selected_genres = st.multiselect(
        "Select Genres for Comparison", sorted(df["genre"].unique()),
        default=sorted(df["genre"].unique()), key="comparison_genres"
    )
    comparison_df = df[
        df["platform"].isin(selected_platforms) &
        df["genre"].isin(selected_genres)
    ]

    st.subheader("🎮 Sales by Genre")
    genre = genre_analysis(comparison_df)
    if not genre.empty:
        fig5 = px.bar(genre, x="genre", y="total_sales", text_auto=True,
                      title="Sales by Genre")
        st.plotly_chart(fig5, use_container_width=True)

    st.subheader("⭐ Average Rating by Platform")
    platform_rating = platform_rating_comparison(comparison_df)
    if not platform_rating.empty:
        fig6 = px.bar(platform_rating, x="platform", y="average_rating",
                      text_auto=True, title="Average Rating by Platform")
        st.plotly_chart(fig6, use_container_width=True)

    st.subheader("📊 Rating Category Distribution")
    ratings = rating_analysis(comparison_df)
    if not ratings.empty:
        fig7 = px.pie(ratings, names="rating_category", values="games",
                      title="Rating Category Distribution")
        st.plotly_chart(fig7, use_container_width=True)

    st.subheader("🎮 Genre vs Platform Sales")
    pivot = genre_platform_analysis(comparison_df)
    st.dataframe(pivot, use_container_width=True)


# ============================================================
# DYNAMIC DATA MANAGEMENT
# ============================================================
def show_dataset_manager():
    st.title("🗂️ Manage Gaming Dataset")
    st.write(
        "Make the dashboard dynamic by uploading your own CSV, adding games, "
        "or editing/removing rows. Analytics pages will use the updated data."
    )

    active_df = st.session_state.gaming_df
    st.metric("Games in active dataset", len(active_df))

    st.subheader("1. Upload your own dataset")
    st.write(
        "Your CSV must contain these columns: `game_name`, `platform`, `genre`, "
        "`year`, `sales`, and `rating`. `critic_score` and `user_score` are optional."
    )
    template = create_gaming_data().to_csv(index=False)
    st.download_button(
        "⬇️ Download CSV template",
        data=template,
        file_name="gaming_data_template.csv",
        mime="text/csv",
        key="download_template"
    )

    uploaded_file = st.file_uploader(
        "Choose a gaming CSV file",
        type=["csv"],
        key="gaming_csv_upload"
    )
    if st.button("Load uploaded CSV", use_container_width=True, key="load_csv_button"):
        if uploaded_file is None:
            st.warning("Choose a CSV file first.")
        else:
            try:
                raw_uploaded_df = pd.read_csv(uploaded_file)
                processed_df = process_data(raw_uploaded_df)
                if processed_df.empty:
                    st.warning("The uploaded CSV has no rows. The active dataset was not changed.")
                else:
                    st.session_state.gaming_df = processed_df
                    st.session_state.dataset_revision += 1
                    st.session_state.flash_message = (
                        f"Loaded {len(processed_df)} game records. The analytics now use your uploaded data."
                    )
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not load this CSV: {exc}")

    st.markdown("---")
    st.subheader("2. Add a game record")
    with st.form("add_game_record_form", clear_on_submit=True):
        game_name = st.text_input("Game name", placeholder="e.g. Cyberpunk 2077")
        c1, c2 = st.columns(2)
        with c1:
            platform = st.text_input("Platform", placeholder="e.g. PC, PlayStation 5")
            genre = st.text_input("Genre", placeholder="e.g. RPG, Action")
            year = st.number_input("Release year", min_value=1970, max_value=2100, value=2025, step=1)
        with c2:
            sales = st.number_input("Sales (use a consistent unit)", min_value=0.0, value=1.0, step=0.5)
            rating = st.slider("Rating out of 10", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
            critic_score = st.number_input("Critic score (optional, 0–100)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        add_submitted = st.form_submit_button("Add game", use_container_width=True)

    if add_submitted:
        if not game_name.strip() or not platform.strip() or not genre.strip():
            st.error("Please fill in the game name, platform, and genre.")
        else:
            new_record = pd.DataFrame([{
                "game_name": game_name.strip(),
                "platform": platform.strip(),
                "genre": genre.strip(),
                "year": int(year),
                "sales": float(sales),
                "rating": float(rating),
                "critic_score": float(critic_score) if critic_score else np.nan,
                "user_score": np.nan,
            }])
            base_columns = ["game_name", "platform", "genre", "year", "sales", "rating", "critic_score", "user_score"]
            current_base = st.session_state.gaming_df.reindex(columns=base_columns)
            combined = pd.concat([current_base, new_record], ignore_index=True)
            st.session_state.gaming_df = process_data(combined)
            st.session_state.dataset_revision += 1
            st.session_state.flash_message = f"Added '{game_name.strip()}'. Analytics have been recalculated."
            st.rerun()

    st.markdown("---")
    st.subheader("3. Edit or remove existing records")
    st.caption("Make changes in the table, use the table's row controls to add/remove rows, then save changes.")
    base_columns = ["game_name", "platform", "genre", "year", "sales", "rating", "critic_score", "user_score"]
    current_base = st.session_state.gaming_df.reindex(columns=base_columns)
    edited_df = st.data_editor(
        current_base,
        num_rows="dynamic",
        use_container_width=True,
        key=f"editable_gaming_table_{st.session_state.dataset_revision}"
    )
    if st.button("Save table changes", use_container_width=True, key="save_table_changes"):
        try:
            st.session_state.gaming_df = process_data(edited_df)
            st.session_state.dataset_revision += 1
            st.session_state.flash_message = "Dataset changes saved for this active app session."
            st.rerun()
        except Exception as exc:
            st.error(f"Could not save the table changes: {exc}")

    st.markdown("---")
    st.subheader("4. Export current dataset")
    export_df = st.session_state.gaming_df
    st.download_button(
        "⬇️ Download current dataset as CSV",
        data=export_df.to_csv(index=False),
        file_name="gaming_dataset_current.csv",
        mime="text/csv",
        key="download_current_dataset"
    )

    if st.button("Reset to sample gaming data", key="reset_sample_data"):
        st.session_state.gaming_df = process_data()
        st.session_state.dataset_revision += 1
        st.session_state.flash_message = "Reset to the built-in sample dataset."
        st.rerun()

    st.info(
        "Note: uploaded/edited data is kept in Streamlit session state, not a permanent database. "
        "Download the current CSV before ending the session if you need to keep your changes."
    )


# ============================================================
# STREAMLIT NAVIGATION
# ============================================================
st.sidebar.title("🎮 Gaming Analytics")

if st.session_state.logged_in:
    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Homepage", "🗂️ Manage Dataset", "📊 Main Content", "🎯 Comparative Analytics"]
    )
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        logout()
else:
    page = st.sidebar.radio("Navigation", ["🏠 Homepage", "🔐 Sign In"])


# ============================================================
# PAGE ROUTING
# ============================================================
if page == "🏠 Homepage":
    show_homepage()
elif page == "🔐 Sign In":
    show_login()
elif page == "🗂️ Manage Dataset":
    if st.session_state.logged_in:
        show_dataset_manager()
    else:
        st.warning("Please sign in to manage the dataset.")
        show_login()
elif page == "📊 Main Content":
    if st.session_state.logged_in:
        show_main_content()
    else:
        st.warning("Please sign in to access Module 3.")
        show_login()
elif page == "🎯 Comparative Analytics":
    if st.session_state.logged_in:
        show_comparative_content()
    else:
        st.warning("Please sign in to access Module 4.")
        show_login()


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "Gaming Industry Trends & Comparative Analytics | "
    "Akash Bhardwaj & Keshav Kumar Sahni"
)
