import streamlit as st
import pandas as pd
import plotly.express as px
import mysql.connector
from mysql.connector import Error as MySQLError
import bcrypt
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from theme import apply_theme, apply_plotly_defaults, render_table

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

# ---------- Plotly visual defaults (presentation only) ----------
apply_plotly_defaults(st.session_state.get("dark_mode", False))

st.set_page_config(
    page_title="InsightHub | Smart Data Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DATABASE CONFIG
# ============================================================
# Credentials are read from .streamlit/secrets.toml if present:
#
#   db_host = "localhost"
#   db_user = "root"
#   db_password = "your-password"
#   db_name = "insighthub"
#
# Falling back to the original hardcoded values keeps this working
# out of the box, but for real deployments PLEASE move the password
# into secrets.toml instead of leaving it in source control.


def get_db_config():
    try:
        return {
            "host": st.secrets["db_host"],
            "user": st.secrets["db_user"],
            "password": st.secrets["db_password"],
            "database": st.secrets["db_name"],
        }
    except Exception:
        return {
            "host": "localhost",
            "user": "root",
            "password": "krish@1235",
            "database": "insighthub",
        }


def connect_db():
    return mysql.connector.connect(**get_db_config())


def db_available():
    try:
        conn = connect_db()
        conn.close()
        return True
    except MySQLError:
        return False


# ============================================================
# AUTH: SCHEMA + HELPERS
# ============================================================

def init_auth_schema():
    """Creates the users table and extends upload_history with user_id.
    Safe to call repeatedly."""
    try:
        conn = connect_db()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(120) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS upload_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                file_name VARCHAR(255),
                rows_count INT,
                columns_count INT,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add user_id column to upload_history if it doesn't already exist
        cursor.execute("""
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_schema = DATABASE()
              AND table_name = 'upload_history'
              AND column_name = 'user_id'
        """)
        has_user_id = cursor.fetchone()[0]

        if not has_user_id:
            cursor.execute("ALTER TABLE upload_history ADD COLUMN user_id INT NULL")

        conn.commit()
        cursor.close()
        conn.close()
    except MySQLError as e:
        st.session_state["_db_error"] = str(e)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


def create_user(username, email, password):
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
            (username, email, hash_password(password)),
        )
        conn.commit()
        return True, "Account created successfully."
    except MySQLError as e:
        if "Duplicate entry" in str(e):
            return False, "That username or email is already registered."
        return False, f"Could not create account: {e}"
    finally:
        cursor.close()
        conn.close()


def authenticate_user(identifier, password):
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM users WHERE username = %s OR email = %s",
        (identifier, identifier),
    )
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if user and verify_password(password, user["password_hash"]):
        return user
    return None


def save_upload_history(file_name, rows, columns):
    try:
        conn = connect_db()
        cursor = conn.cursor()
        user_id = st.session_state.get("user", {}).get("id")
        cursor.execute(
            """
            INSERT INTO upload_history (file_name, rows_count, columns_count, user_id)
            VALUES (%s, %s, %s, %s)
            """,
            (file_name, rows, columns, user_id),
        )
        conn.commit()
        cursor.close()
        conn.close()
    except MySQLError:
        pass  # Don't block the app if history logging fails


# ============================================================
# PDF REPORT
# ============================================================

def generate_pdf(data):

    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Table,
        TableStyle,
        Spacer,
        PageBreak
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from datetime import datetime
    import math

    # ============================================================
    # COLORS — INSIGHTHUB YELLOW + WHITE THEME
    # ============================================================

    YELLOW = colors.HexColor("#FFD60A")
    SOFT_YELLOW = colors.HexColor("#FFF3A3")
    TINT_YELLOW = colors.HexColor("#FFFBD0")
    BLACK = colors.HexColor("#111111")
    DARK_GREY = colors.HexColor("#333333")
    GREY = colors.HexColor("#666666")
    LIGHT_GREY = colors.HexColor("#E9E9E9")
    PAGE_BG = colors.HexColor("#FAFAFA")
    WHITE = colors.white
    GREEN = colors.HexColor("#2E7D32")
    RED = colors.HexColor("#C62828")

    # ============================================================
    # DATASET INFORMATION
    # ============================================================

    rows = int(data.shape[0])
    cols = int(data.shape[1])

    missing = int(data.isnull().sum().sum())
    duplicates = int(data.duplicated().sum())

    numeric_cols = data.select_dtypes(include="number").columns.tolist()
    text_cols = data.select_dtypes(exclude="number").columns.tolist()

    numeric_count = len(numeric_cols)
    text_count = len(text_cols)

    # Same basic quality idea used by InsightHub
    quality_score = 100
    quality_score -= missing
    quality_score -= duplicates * 2
    quality_score = max(0, min(100, quality_score))

    # ============================================================
    # PDF DOCUMENT
    # ============================================================

    doc = SimpleDocTemplate(
        "InsightHub_Report.pdf",
        pagesize=(8.27 * inch, 11.69 * inch),
        rightMargin=42,
        leftMargin=42,
        topMargin=55,
        bottomMargin=45,
        title="InsightHub Analytics Report",
        author="InsightHub"
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=25,
        leading=30,
        textColor=BLACK,
        alignment=TA_LEFT,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=GREY
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=BLACK,
        spaceBefore=8,
        spaceAfter=10
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=DARK_GREY
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=GREY
    )

    card_value_style = ParagraphStyle(
        "CardValue",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=20,
        textColor=BLACK,
        alignment=TA_LEFT
    )

    card_label_style = ParagraphStyle(
        "CardLabel",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=GREY
    )

    elements = []

    # ============================================================
    # HEADER / FOOTER
    # ============================================================

    def draw_header_footer(canvas, doc):

        canvas.saveState()

        page_width, page_height = doc.pagesize

        # Top yellow line
        canvas.setFillColor(YELLOW)
        canvas.rect(
            0,
            page_height - 7,
            page_width,
            7,
            fill=1,
            stroke=0
        )

        # Header
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(BLACK)
        canvas.drawString(
            42,
            page_height - 30,
            "INSIGHTHUB"
        )

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(GREY)
        canvas.drawRightString(
            page_width - 42,
            page_height - 30,
            "Smart Data Analytics Dashboard"
        )

        # Footer line
        canvas.setStrokeColor(LIGHT_GREY)
        canvas.line(
            42,
            31,
            page_width - 42,
            31
        )

        # Footer text
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GREY)

        canvas.drawString(
            42,
            19,
            "InsightHub • Smart Data Analytics"
        )

        canvas.drawRightString(
            page_width - 42,
            19,
            f"Page {doc.page}"
        )

        canvas.restoreState()

    # ============================================================
    # PAGE 1 — COVER + EXECUTIVE SUMMARY
    # ============================================================

    elements.append(Spacer(1, 35))

    elements.append(
        Paragraph(
            "DATA ANALYTICS<br/>REPORT",
            title_style
        )
    )

    elements.append(
        Paragraph(
            "A professional dataset summary generated by InsightHub",
            subtitle_style
        )
    )

    elements.append(Spacer(1, 20))

    # Dataset name
    elements.append(
        Table(
            [[
                Paragraph(
                    "<b>Dataset</b><br/>Current InsightHub Dataset",
                    body_style
                ),
                Paragraph(
                    f"<b>Generated</b><br/>{datetime.now().strftime('%d %B %Y, %I:%M %p')}",
                    body_style
                )
            ]],
            colWidths=[3.35 * inch, 3.35 * inch],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), TINT_YELLOW),
                ("BOX", (0, 0), (-1, -1), 0.7, LIGHT_GREY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ])
        )
    )

    elements.append(Spacer(1, 20))

    # ============================================================
    # KPI CARDS
    # ============================================================

    cards = [
        [
            Paragraph(str(rows), card_value_style),
            Paragraph(str(cols), card_value_style),
            Paragraph(str(missing), card_value_style),
            Paragraph(str(duplicates), card_value_style)
        ],
        [
            Paragraph("TOTAL ROWS", card_label_style),
            Paragraph("COLUMNS", card_label_style),
            Paragraph("MISSING VALUES", card_label_style),
            Paragraph("DUPLICATE ROWS", card_label_style)
        ]
    ]

    card_table = Table(
        cards,
        colWidths=[
            1.675 * inch,
            1.675 * inch,
            1.675 * inch,
            1.675 * inch
        ]
    )

    card_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WHITE),
        ("BOX", (0, 0), (-1, -1), 0.7, LIGHT_GREY),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, LIGHT_GREY),

        ("BACKGROUND", (0, 0), (0, 0), TINT_YELLOW),
        ("BACKGROUND", (1, 0), (1, 0), TINT_YELLOW),
        ("BACKGROUND", (2, 0), (2, 0), TINT_YELLOW),
        ("BACKGROUND", (3, 0), (3, 0), TINT_YELLOW),

        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, 0), 13),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ("TOPPADDING", (0, 1), (-1, 1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 12),
    ]))

    elements.append(card_table)

    elements.append(Spacer(1, 22))

    # ============================================================
    # DATA QUALITY
    # ============================================================

    elements.append(
        Paragraph("Data Quality", section_style)
    )

    quality_color = GREEN if quality_score >= 80 else RED

    quality_text = (
        "Excellent"
        if quality_score >= 90
        else "Good"
        if quality_score >= 75
        else "Needs Attention"
    )

    quality_bar = Table(
        [
            [
                Paragraph(
                    f"<b>{quality_score}/100</b>",
                    ParagraphStyle(
                        "qscore",
                        parent=body_style,
                        fontSize=15,
                        textColor=quality_color
                    )
                ),
                Paragraph(
                    f"<b>{quality_text}</b><br/>Overall dataset quality based on missing and duplicate records.",
                    body_style
                )
            ]
        ],
        colWidths=[1.1 * inch, 5.6 * inch]
    )

    quality_bar.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), SOFT_YELLOW),
        ("BACKGROUND", (1, 0), (1, 0), PAGE_BG),
        ("BOX", (0, 0), (-1, -1), 0.7, LIGHT_GREY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
    ]))

    elements.append(quality_bar)

    elements.append(Spacer(1, 25))

    elements.append(
        Paragraph(
            "<b>Executive Summary</b>",
            section_style
        )
    )

    elements.append(
        Paragraph(
            f"This report provides a structured overview of the uploaded dataset. "
            f"The dataset contains <b>{rows}</b> records across <b>{cols}</b> columns, "
            f"including <b>{numeric_count}</b> numeric columns and "
            f"<b>{text_count}</b> text/categorical columns. "
            f"The report also highlights data quality, statistical information, "
            f"sample records and useful recommendations for further analysis.",
            body_style
        )
    )

    # ============================================================
    # PAGE 2 — DATASET PROFILE
    # ============================================================

    elements.append(PageBreak())

    elements.append(
        Paragraph("Dataset Profile", title_style)
    )

    elements.append(
        Paragraph(
            "Structure and quality overview of the current dataset.",
            subtitle_style
        )
    )

    elements.append(Spacer(1, 18))

    profile = [
        ["Dataset Property", "Value"],
        ["Total Records", rows],
        ["Total Columns", cols],
        ["Numeric Columns", numeric_count],
        ["Text / Categorical Columns", text_count],
        ["Missing Values", missing],
        ["Duplicate Rows", duplicates],
        ["Data Quality Score", f"{quality_score}/100"],
    ]

    profile_table = Table(
        profile,
        colWidths=[3.8 * inch, 2.9 * inch],
        repeatRows=1
    )

    profile_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLACK),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

        ("BACKGROUND", (0, 1), (-1, -1), WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PAGE_BG]),

        ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_GREY),

        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 1), (-1, -1), DARK_GREY),

        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))

    elements.append(profile_table)

    elements.append(Spacer(1, 22))

    # ============================================================
    # COLUMN PROFILE
    # ============================================================

    elements.append(
        Paragraph("Column Profile", section_style)
    )

    column_profile = [
        ["Column", "Data Type", "Non-Null %"]
    ]

    for col in data.columns:

        non_null_percent = (
            data[col].notna().mean() * 100
            if rows > 0
            else 0
        )

        column_profile.append([
            Paragraph(str(col), small_style),
            Paragraph(str(data[col].dtype), small_style),
            f"{non_null_percent:.1f}%"
        ])

    column_table = Table(
        column_profile,
        colWidths=[3.4 * inch, 1.8 * inch, 1.5 * inch],
        repeatRows=1
    )

    column_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), YELLOW),
        ("TEXTCOLOR", (0, 0), (-1, 0), BLACK),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PAGE_BG]),
        ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_GREY),

        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    elements.append(column_table)

    # ============================================================
    # PAGE 3 — STATISTICAL ANALYSIS
    # ============================================================

    elements.append(PageBreak())

    elements.append(
        Paragraph("Statistical Analysis", title_style)
    )

    elements.append(
        Paragraph(
            "Key numerical statistics extracted from the dataset.",
            subtitle_style
        )
    )

    elements.append(Spacer(1, 18))

    if numeric_count > 0:

        stats_rows = [
            ["Column", "Count", "Mean", "Median", "Minimum", "Maximum"]
        ]

        for col in numeric_cols:

            series = pd.to_numeric(data[col], errors="coerce").dropna()

            if len(series) == 0:
                continue

            stats_rows.append([
                Paragraph(str(col), small_style),
                f"{len(series):,}",
                f"{series.mean():.2f}",
                f"{series.median():.2f}",
                f"{series.min():.2f}",
                f"{series.max():.2f}"
            ])

        if len(stats_rows) > 1:

            stats_table = Table(
                stats_rows,
                colWidths=[
                    1.55 * inch,
                    0.75 * inch,
                    1.05 * inch,
                    1.05 * inch,
                    1.05 * inch,
                    1.05 * inch
                ],
                repeatRows=1
            )

            stats_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), BLACK),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PAGE_BG]),
                ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_GREY),

                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),

                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]))

            elements.append(stats_table)

    else:

        elements.append(
            Paragraph(
                "No numeric columns were found in this dataset, so statistical analysis is not available.",
                body_style
            )
        )

    elements.append(Spacer(1, 25))

    # ============================================================
    # KEY INSIGHTS
    # ============================================================

    elements.append(
        Paragraph("Key Insights", section_style)
    )

    insights = []

    if numeric_count > 0:

        valid_numeric = []

        for col in numeric_cols:

            series = pd.to_numeric(data[col], errors="coerce").dropna()

            if len(series) > 0:
                valid_numeric.append((col, series))

        if valid_numeric:

            # Highest mean
            highest_mean = max(
                valid_numeric,
                key=lambda x: x[1].mean()
            )

            insights.append(
                f"<b>Highest average:</b> {highest_mean[0]} "
                f"has an average value of {highest_mean[1].mean():.2f}."
            )

            # Highest maximum
            highest_max = max(
                valid_numeric,
                key=lambda x: x[1].max()
            )

            insights.append(
                f"<b>Highest observed value:</b> {highest_max[0]} "
                f"reaches {highest_max[1].max():.2f}."
            )

            # Correlation
            if numeric_count >= 2:

                corr = data[numeric_cols].corr()

                corr_pairs = []

                for i in range(len(numeric_cols)):
                    for j in range(i + 1, len(numeric_cols)):

                        value = corr.iloc[i, j]

                        if not pd.isna(value):
                            corr_pairs.append(
                                (
                                    abs(value),
                                    value,
                                    numeric_cols[i],
                                    numeric_cols[j]
                                )
                            )

                if corr_pairs:

                    _, corr_value, col_a, col_b = max(
                        corr_pairs,
                        key=lambda x: x[0]
                    )

                    insights.append(
                        f"<b>Strongest relationship:</b> "
                        f"{col_a} and {col_b} have a correlation of "
                        f"{corr_value:.2f}."
                    )

    if not insights:

        insights.append(
            "<b>Dataset structure:</b> The dataset is primarily categorical/text-based, "
            "so categorical analysis and visual comparisons are recommended."
        )

    insight_data = []

    for i, insight in enumerate(insights, start=1):
        insight_data.append([
            Paragraph(f"<b>{i}</b>", body_style),
            Paragraph(insight, body_style)
        ])

    insight_table = Table(
        insight_data,
        colWidths=[0.45 * inch, 6.25 * inch]
    )

    insight_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), SOFT_YELLOW),
        ("BACKGROUND", (1, 0), (1, -1), WHITE),

        ("BOX", (0, 0), (-1, -1), 0.7, LIGHT_GREY),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, LIGHT_GREY),

        ("VALIGN", (0, 0), (-1, -1), "TOP"),

        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))

    elements.append(insight_table)

    # ============================================================
    # PAGE 4 — DATASET PREVIEW
    # ============================================================

    elements.append(PageBreak())

    elements.append(
        Paragraph("Dataset Preview", title_style)
    )

    elements.append(
        Paragraph(
            "First 10 records from the uploaded dataset.",
            subtitle_style
        )
    )

    elements.append(Spacer(1, 18))

    preview = data.head(10).copy()

    # Limit columns so the PDF remains readable
    max_preview_columns = 7

    preview_columns = preview.columns.tolist()

    if len(preview_columns) > max_preview_columns:
        visible_columns = preview_columns[:max_preview_columns]
        hidden_count = len(preview_columns) - max_preview_columns
        preview = preview[visible_columns]
    else:
        hidden_count = 0

    preview_data = [
        [Paragraph(f"<b>{str(col)}</b>", small_style) for col in preview.columns]
    ]

    for _, row in preview.iterrows():

        formatted_row = []

        for value in row:

            if pd.isna(value):
                value = "—"
            else:
                value = str(value)

            # Prevent extremely long text from breaking the table
            if len(value) > 35:
                value = value[:32] + "..."

            formatted_row.append(
                Paragraph(value, small_style)
            )

        preview_data.append(formatted_row)

    if len(preview_data) > 1:

        preview_table = Table(
            preview_data,
            repeatRows=1
        )

        preview_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), YELLOW),
            ("TEXTCOLOR", (0, 0), (-1, 0), BLACK),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PAGE_BG]),
            ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_GREY),

            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))

        elements.append(preview_table)

    if hidden_count > 0:

        elements.append(Spacer(1, 8))

        elements.append(
            Paragraph(
                f"+ {hidden_count} additional column(s) not shown in this preview for readability.",
                small_style
            )
        )

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(
            "<b>Preview Note</b>",
            section_style
        )
    )

    elements.append(
        Paragraph(
            "The preview displays a maximum of 10 records and 7 columns "
            "to keep the report clean and readable.",
            body_style
        )
    )

    # ============================================================
    # PAGE 5 — RECOMMENDATIONS
    # ============================================================

    elements.append(PageBreak())

    elements.append(
        Paragraph("InsightHub Recommendations", title_style)
    )

    elements.append(
        Paragraph(
            "Suggested next steps based on the current dataset.",
            subtitle_style
        )
    )

    elements.append(Spacer(1, 22))

    recommendations = []

    if missing > 0:
        recommendations.append(
            "<b>Review missing values:</b> "
            f"The dataset contains {missing} missing value(s). "
            "Consider cleaning or imputing them before advanced analysis."
        )
    else:
        recommendations.append(
            "<b>Missing values:</b> "
            "No missing values were detected. The dataset is ready for further analysis."
        )

    if duplicates > 0:
        recommendations.append(
            "<b>Remove duplicates:</b> "
            f"{duplicates} duplicate row(s) were detected. "
            "Review and remove them if they do not represent valid repeated records."
        )
    else:
        recommendations.append(
            "<b>Duplicate check:</b> "
            "No duplicate rows were detected."
        )

    if numeric_count > 0:
        recommendations.append(
            "<b>Explore numerical data:</b> "
            "Use statistical analysis, correlation analysis and numerical charts "
            "to identify important relationships."
        )

    if text_count > 0:
        recommendations.append(
            "<b>Explore categorical data:</b> "
            "Use bar charts, pie charts or comparison visualizations "
            "to understand category distributions."
        )

    recommendations.append(
        "<b>Build visualizations:</b> "
        "Use InsightHub's visualization tools to convert important dataset patterns "
        "into easy-to-understand charts."
    )

    recommendation_data = []

    for i, recommendation in enumerate(recommendations, start=1):

        recommendation_data.append([
            Paragraph(
                f"<b>{i}</b>",
                ParagraphStyle(
                    "RecNumber",
                    parent=body_style,
                    fontSize=12,
                    textColor=BLACK,
                    alignment=TA_CENTER
                )
            ),
            Paragraph(
                recommendation,
                body_style
            )
        ])

    recommendation_table = Table(
        recommendation_data,
        colWidths=[0.55 * inch, 6.15 * inch]
    )

    recommendation_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), YELLOW),
        ("BACKGROUND", (1, 0), (1, -1), WHITE),

        ("BOX", (0, 0), (-1, -1), 0.7, LIGHT_GREY),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, LIGHT_GREY),

        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
    ]))

    elements.append(recommendation_table)

    elements.append(Spacer(1, 35))

    # Final branding card
    final_card = Table(
        [[
            Paragraph(
                "<b>InsightHub</b><br/>"
                "Smart Data Analytics Dashboard<br/><br/>"
                "Turn Data into Insights.",
                ParagraphStyle(
                    "FinalCard",
                    parent=body_style,
                    fontSize=11,
                    leading=16,
                    textColor=BLACK
                )
            )
        ]],
        colWidths=[6.7 * inch]
    )

    final_card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), TINT_YELLOW),
        ("BOX", (0, 0), (-1, -1), 0.8, YELLOW),
        ("LEFTPADDING", (0, 0), (-1, -1), 18),
        ("RIGHTPADDING", (0, 0), (-1, -1), 18),
        ("TOPPADDING", (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
    ]))

    elements.append(final_card)

    # ============================================================
    # BUILD PDF
    # ============================================================

    doc.build(
        elements,
        onFirstPage=draw_header_footer,
        onLaterPages=draw_header_footer
    )

# ============================================================
# GLOBAL STYLE
# ============================================================

apply_theme(st.session_state.get("dark_mode", False))


# ============================================================
# SESSION STATE INIT
# ============================================================

if "authenticated" not in st.session_state:
    # InsightHub opens directly on the dashboard.
    # The default visitor is a guest until they choose Account / Sign In.
    st.session_state["authenticated"] = True
if "user" not in st.session_state:
    st.session_state["user"] = {"id": None, "username": "Guest", "email": None}
if "guest" not in st.session_state:
    st.session_state["guest"] = True
if "show_auth" not in st.session_state:
    st.session_state["show_auth"] = False


# ============================================================
# AUTH SCREEN
# ============================================================

def render_auth_screen():
    # Decorative background glow
    st.markdown('<div class="auth-bg-blob-a"></div><div class="auth-bg-blob-b"></div>', unsafe_allow_html=True)

    # Easy escape route in case the user opened Account / Sign In by mistake.
    back_col, _ = st.columns([1.2, 5])
    with back_col:
        if st.button("← Back to Dashboard", key="auth_back_dashboard", use_container_width=True):
            st.session_state["show_auth"] = False
            st.rerun()

    connected = db_available()

    left_pad, left, right, right_pad = st.columns([0.4, 5, 5, 0.4], gap="large")

    # ---------------- LEFT: brand panel ----------------
    with left:
        st.markdown("""
        <div class="brand-panel">
            <div class="blob blob-1"></div>
            <div class="blob blob-2"></div>
            <div class="blob blob-3"></div>
            <div class="brand-badge">📊</div>
            <h1>InsightHub</h1>
            <p class="brand-tag">Your data, understood at a glance. Upload a
            spreadsheet and walk out with clean data, real insight, and a
            report worth sharing.</p>
            <div class="brand-feature"><span class="fi">📂</span><span>Upload CSV & Excel in seconds</span></div>
            <div class="brand-feature"><span class="fi">🧹</span><span>Auto-clean missing values & duplicates</span></div>
            <div class="brand-feature"><span class="fi">📈</span><span>Smart, auto-recommended visualizations</span></div>
            <div class="brand-feature"><span class="fi">📄</span><span>One-click PDF analytics reports</span></div>
            <div class="brand-footnote">Trusted, private, and yours — every dataset stays
            tied to your account.</div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- RIGHT: auth card ----------------
    with right:
        with st.container(border=True):

            if not connected:
                st.warning(
                    "⚠ Can't reach the MySQL database right now, so account "
                    "sign-in is unavailable. You can continue as a guest instead."
                )

            tab_signin, tab_signup = st.tabs(["🔐 Sign In", "🆕 Create Account"])

            with tab_signin:
                st.markdown("""
                <div class="auth-heading">
                    <h2>Welcome back</h2>
                    <p>Sign in to pick up right where you left off.</p>
                </div>
                """, unsafe_allow_html=True)

                with st.form("signin_form"):
                    identifier = st.text_input("Username or Email", placeholder="you@example.com")
                    password = st.text_input("Password", type="password", placeholder="••••••••")
                    submitted = st.form_submit_button("Sign In →", disabled=not connected)

                if submitted:
                    if not identifier or not password:
                        st.error("Please enter both your username/email and password.")
                    else:
                        user = authenticate_user(identifier, password)
                        if user:
                            st.session_state["authenticated"] = True
                            st.session_state["guest"] = False
                            st.session_state["show_auth"] = False
                            st.session_state["user"] = {
                                "id": user["id"],
                                "username": user["username"],
                                "email": user["email"],
                            }
                            st.rerun()
                        else:
                            st.error("Incorrect username/email or password.")

            with tab_signup:
                st.markdown("""
                <div class="auth-heading">
                    <h2>Create your account</h2>
                    <p>Takes less than a minute — no credit card, no spam.</p>
                </div>
                """, unsafe_allow_html=True)

                with st.form("signup_form"):
                    new_username = st.text_input("Choose a username", placeholder="krish_k")
                    new_email = st.text_input("Email", placeholder="you@example.com")
                    new_password = st.text_input("Choose a password", type="password", placeholder="At least 6 characters")
                    confirm_password = st.text_input("Confirm password", type="password", placeholder="Re-enter your password")
                    signup_submitted = st.form_submit_button("Create Account →", disabled=not connected)

                if signup_submitted:
                    if not new_username or not new_email or not new_password:
                        st.error("Please fill in every field.")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters long.")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = create_user(new_username, new_email, new_password)
                        if ok:
                            st.success(f"✅ {msg} You can now sign in from the Sign In tab.")
                        else:
                            st.error(msg)

            st.markdown('<div class="guest-divider">OR</div>', unsafe_allow_html=True)

            st.markdown('<div class="guest-btn">', unsafe_allow_html=True)
            if st.button("👤 Continue as Guest"):
                st.session_state["authenticated"] = True
                st.session_state["guest"] = True
                st.session_state["show_auth"] = False
                st.session_state["user"] = {"id": None, "username": "Guest", "email": None}
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        "<div style='text-align:center; margin-top:22px;'>",
        unsafe_allow_html=True,
    )
    st.caption("© 2026 InsightHub | Developed using Streamlit & Python")
    st.markdown("</div>", unsafe_allow_html=True)


# Authentication is opened only when the visitor chooses Account / Sign In.
# The normal startup experience is the dashboard as a guest.
if st.session_state.get("show_auth", False):
    init_auth_schema()
    render_auth_screen()
    st.stop()


# ============================================================
# AUTHENTICATED APP
# ============================================================

user = st.session_state["user"]
is_guest = st.session_state["guest"]

initials = "GU" if is_guest else "".join([w[0] for w in user["username"][:2].upper()])

st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="sidebar-brand-icon">
        <svg viewBox="0 0 64 64" width="34" height="34" aria-hidden="true">
            <defs><linearGradient id="ihLogo" x1="0" y1="1" x2="1" y2="0"><stop offset="0%" stop-color="#E6A300"/><stop offset="55%" stop-color="#FFC928"/><stop offset="100%" stop-color="#172033"/></linearGradient></defs>
            <path d="M8 44 C19 42 22 30 31 30 C39 30 43 38 56 20" fill="none" stroke="url(#ihLogo)" stroke-width="5" stroke-linecap="round"/>
            <rect x="12" y="35" width="6" height="13" rx="2" fill="#E6A300"/><rect x="23" y="28" width="6" height="20" rx="2" fill="#F5B91D"/><rect x="34" y="20" width="6" height="28" rx="2" fill="#FFD447"/>
            <path d="M47 11 L47 27 L58 27" fill="none" stroke="#172033" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
    </div>
    <div><div class="sidebar-brand-title">InsightHub</div><div class="sidebar-brand-subtitle">Smart Data Analytics</div></div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div style='font-size:.67rem;font-weight:800;letter-spacing:.18em;color:#6A5011;margin:0 6px 4px;'>MENU</div>", unsafe_allow_html=True)

nav_items = [
    "Home", "Upload Dataset", "Data Cleaning", "Data Analysis",
    "Visualizations", "Predictive Analytics", "Upload History", "Reports", "About",
]

if st.session_state.get("page") not in nav_items:
    st.session_state["page"] = "Home"

for nav_item in nav_items:
    if st.sidebar.button(nav_item, key="nav_" + nav_item, type="primary" if st.session_state["page"] == nav_item else "secondary", use_container_width=True):
        st.session_state["page"] = nav_item
        st.rerun()

page = st.session_state["page"]

st.sidebar.markdown("<div style='flex:1;min-height:12px'></div>", unsafe_allow_html=True)

# Account access is optional: the dashboard is available immediately as Guest.
if is_guest:
    if st.sidebar.button("👤 Account / Sign In", key="open_account", use_container_width=True):
        st.session_state["show_auth"] = True
        st.rerun()
else:
    if st.sidebar.button("👤 Account", key="open_account", use_container_width=True):
        st.session_state["show_auth"] = True
        st.rerun()

with st.sidebar.expander("⚙️ Settings", expanded=False):
    st.caption("Account")
    st.toggle(
        "🌙 Dark mode",
        value=st.session_state.get("dark_mode", False),
        key="dark_toggle",
        on_change=lambda: st.session_state.update(dark_mode=st.session_state["dark_toggle"]),
    )
    st.markdown(f"**{user['username']}**")
    st.caption(user["email"] if not is_guest else "Guest session")
    if st.button("🚪 Log Out", key="settings_logout"):
        # Logging out returns the visitor to the public guest dashboard.
        st.session_state["authenticated"] = True
        st.session_state["guest"] = True
        st.session_state["show_auth"] = False
        st.session_state["user"] = {"id": None, "username": "Guest", "email": None}
        st.rerun()

st.sidebar.markdown(f"""
<div class="user-chip sidebar-user-bottom">
    <div class="user-avatar">{initials[:2]}</div>
    <div class="user-meta"><div class="u-name">{user['username']}</div><div class="u-role">{'Guest session' if is_guest else user['email']}</div></div>
</div>
""", unsafe_allow_html=True)


# ---------------- TOP BAR ----------------
st.markdown(f"""
<div class="app-topbar">
    <div class="app-search"><span class="app-search-icon">⌕</span><span>Search datasets, features, or help...</span></div>
    <div class="app-top-actions">
        <div class="top-action">🔔</div>
        <div class="top-profile"><div class="top-avatar">{initials[:2]}</div><div><small>Welcome</small><strong>{user['username']}</strong></div><span>⌄</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- HOME ----------------

if page == "Home":

    if "data" in st.session_state:

        data = st.session_state["data"]

        st.markdown(f"""
        <div class="hero hero-compact">
            <h1>Welcome back, {user['username']} 👋</h1>
            <p>Upload, clean, analyze, and visualize your data — all in one place.</p>
        </div>
        """, unsafe_allow_html=True)

        row_count = data.shape[0]
        col_count = data.shape[1]
        missing = data.isnull().sum().sum()
        duplicate = data.duplicated().sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📄 Rows", row_count)
        col2.metric("📑 Columns", col_count)
        col3.metric("⚠ Missing", missing)
        col4.metric("🔁 Duplicates", duplicate)

        st.divider()

        left, right = st.columns([2, 1])

        with left:
            st.subheader("📋 Dataset Preview")
            render_table(data.head(10))

        with right:
            st.subheader("📌 Dataset Info")
            st.write(f"**Rows:** {row_count}")
            st.write(f"**Columns:** {col_count}")

            numeric = len(data.select_dtypes(include="number").columns)
            text = len(data.select_dtypes(include="object").columns)

            st.write(f"**Numeric Columns:** {numeric}")
            st.write(f"**Text Columns:** {text}")
            st.write(f"**Missing Values:** {missing}")

    else:

        st.markdown("""
        <div class="hero">
            <div class="hero-copy">
                <div class="hero-kicker">WELCOME BACK</div>
                <h1>Welcome to <span class="accent">InsightHub</span> 👋</h1>
                <p>Upload, clean, analyze, and visualize your data — all in one place.</p>
            </div>
            <div class="hero-visual">
                <svg viewBox="0 0 390 155" width="390" height="155" aria-hidden="true">
                    <defs>
                        <linearGradient id="heroWave" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#FFF2BE"/><stop offset="1" stop-color="#F4C43E" stop-opacity=".45"/></linearGradient>
                    </defs>
                    <path d="M55 150 C120 95 128 45 198 38 C255 32 274 90 390 18 L390 155 L55 155Z" fill="url(#heroWave)" opacity=".62"/>
                    <g class="screen">
                        <rect x="182" y="26" width="128" height="98" rx="9" fill="#19253A" transform="rotate(-5 246 75)"/>
                        <rect x="189" y="34" width="114" height="82" rx="5" fill="#FFFDF8" transform="rotate(-5 246 75)"/>
                        <rect x="199" y="47" width="48" height="6" rx="3" fill="#F1D77A" transform="rotate(-5 246 75)"/>
                        <rect x="201" y="88" width="12" height="20" rx="2" fill="#FFD24B" transform="rotate(-5 246 75)"/>
                        <rect x="220" y="76" width="12" height="32" rx="2" fill="#F3B318" transform="rotate(-5 246 75)"/>
                        <rect x="239" y="62" width="12" height="46" rx="2" fill="#E5A200" transform="rotate(-5 246 75)"/>
                        <rect x="258" y="70" width="12" height="38" rx="2" fill="#FFD447" transform="rotate(-5 246 75)"/>
                        <circle cx="278" cy="56" r="16" fill="#F6B719" transform="rotate(-5 246 75)"/>
                    </g>
                    <rect x="301" y="50" width="58" height="40" rx="8" fill="#FFFFFF" stroke="#F0E5C2"/>
                    <rect x="312" y="60" width="35" height="5" rx="2.5" fill="#E5A200"/>
                    <rect x="312" y="70" width="27" height="5" rx="2.5" fill="#DDE1E8"/>
                    <text x="319" y="128" font-size="12" font-family="cursive" font-weight="700" fill="#6B4D00">Turn</text>
                    <text x="319" y="142" font-size="12" font-family="cursive" font-weight="700" fill="#6B4D00">Data into</text>
                    <text x="319" y="156" font-size="12" font-family="cursive" font-weight="700" fill="#6B4D00">Insights</text>
                </svg>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info("Upload a CSV or Excel dataset to start analyzing your data.")
        st.markdown('<div class="upload-label">Choose a CSV or Excel file</div>', unsafe_allow_html=True)
        home_uploaded_file = st.file_uploader("", type=["csv", "xlsx"], key="home_uploader", label_visibility="collapsed")

        if home_uploaded_file is not None:
            if home_uploaded_file.name.endswith(".csv"):
                home_data = pd.read_csv(home_uploaded_file)
            else:
                home_data = pd.read_excel(home_uploaded_file)
            st.session_state["data"] = home_data
            save_upload_history(home_uploaded_file.name, home_data.shape[0], home_data.shape[1])
            st.success("✅ File Uploaded Successfully!")
            st.rerun()

        st.markdown("""
        <div class="section-heading-row">
            <div><span class="section-accent"></span><h2>Features</h2></div>
            <span class="explore-link">Explore all features →</span>
        </div>
        """, unsafe_allow_html=True)

        feats = [
            ("📂", "Upload", "Bring in CSV or Excel files in seconds."),
            ("🧹", "Clean", "Fix missing values and drop duplicates."),
            ("📊", "Analyze", "Get instant summary stats and insights."),
            ("📈", "Visualize", "Build interactive charts automatically."),
        ]
        cols = st.columns(4, gap="medium")
        for c, (icon, title, desc) in zip(cols, feats):
            with c:
                st.markdown(f"""
                <div class="feat-card">
                    <div class="feat-icon">{icon}</div>
                    <div class="feat-body"><h4>{title}</h4><p>{desc}</p></div>
                    <div class="feat-arrow">→</div>
                </div>
                """, unsafe_allow_html=True)

elif page == "Upload Dataset":

    st.header("📂 Upload Dataset")
    st.markdown('<div class="page-description">Import a CSV or Excel dataset and start exploring it.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.session_state["data"] = data
        save_upload_history(uploaded_file.name, data.shape[0], data.shape[1])

        st.success("✅ File Uploaded Successfully!")

        st.subheader("Dataset Preview")
        render_table(data)

        st.subheader("Dataset Information")
        st.write("Rows:", data.shape[0])
        st.write("Columns:", data.shape[1])

        st.write("### Column Names")
        st.write(", ".join(map(str, data.columns)))

        st.write("### Data Types")
        render_table(data.dtypes.astype(str).rename("Type"), height=300)

elif page == "Data Cleaning":

    st.header("🧹 Data Cleaning")
    st.markdown('<div class="page-description">Identify missing values and duplicates, then prepare a cleaner dataset.</div>', unsafe_allow_html=True)

    if "data" in st.session_state:

        data = st.session_state["data"]

        st.subheader("Original Dataset")
        render_table(data)

        st.subheader("Missing Values")
        render_table(data.isnull().sum().rename("Missing"), height=300)

        st.write(f"Duplicate Rows: {data.duplicated().sum()}")

        c1, c2 = st.columns(2)

        with c1:
            if st.button("Fill Missing Values"):
                cleaned_data = data.copy()

                for col in cleaned_data.select_dtypes(include=["number"]).columns:
                    cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mean())

                for col in cleaned_data.select_dtypes(include=["object"]).columns:
                    if not cleaned_data[col].mode().empty:
                        cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mode()[0])

                st.session_state["data"] = cleaned_data
                st.success("✅ Missing values filled successfully!")
                data = cleaned_data

        with c2:
            if st.button("Remove Duplicate Rows"):
                cleaned_data = st.session_state["data"].drop_duplicates()
                st.session_state["data"] = cleaned_data
                st.success("✅ Duplicate rows removed successfully!")
                data = cleaned_data

        st.subheader("Cleaned Dataset")
        render_table(data)

        csv = data.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Cleaned Dataset",
            data=csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
        )

    else:
        st.warning("⚠ Please upload a dataset first.")

# ==========================================================
# DATA ANALYSIS
# ==========================================================

elif page == "Data Analysis":

    st.header("📊 Data Analysis")
    st.markdown('<div class="page-description">Understand your dataset through quality metrics, statistics, and automated insights.</div>', unsafe_allow_html=True)

    if "data" in st.session_state:

        data = st.session_state["data"]

        st.success("✅ Dataset Loaded Successfully!")

        st.subheader("💡 Smart Data Insights")

        rows = data.shape[0]
        cols = data.shape[1]

        numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = data.select_dtypes(exclude=["number"]).columns.tolist()

        missing_values = data.isnull().sum().sum()
        duplicate_rows = data.duplicated().sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📄 Rows", rows)
        col2.metric("📑 Columns", cols)
        col3.metric("🔢 Numeric", len(numeric_columns))
        col4.metric("📝 Categorical", len(categorical_columns))

        col5, col6 = st.columns(2)
        col5.metric("⚠ Missing Values", missing_values)
        col6.metric("🔁 Duplicate Rows", duplicate_rows)

        st.divider()

        score = 100
        score -= missing_values
        score -= duplicate_rows * 2
        if score < 0:
            score = 0

        st.subheader("📊 Data Quality Score")
        st.progress(score / 100)
        st.success(f"Overall Dataset Quality : {score}/100")

        st.divider()

        if len(numeric_columns) >= 2:
            corr = data[numeric_columns].corr()
            corr_pairs = corr.where(~corr.eq(1)).stack().sort_values(ascending=False)

            if not corr_pairs.empty:
                highest = corr_pairs.index[0]
                value = corr_pairs.iloc[0]

                st.subheader("📈 Highest Correlation")
                st.success(f"{highest[0]} ↔ {highest[1]} ({value:.2f})")

        st.divider()

        st.subheader("🧠 InsightHub Recommendation")

        if missing_values > 0:
            st.warning("Missing values detected. Clean the dataset before visualization.")
        elif duplicate_rows > 0:
            st.warning("Duplicate rows detected. Remove duplicate records.")
        else:
            st.success("Dataset is clean and ready for further analysis.")

        if len(numeric_columns) >= 2:
            st.info("This dataset is suitable for Correlation Analysis and Machine Learning.")
        elif len(categorical_columns) >= 1:
            st.info("This dataset is suitable for Category Comparison.")

        st.divider()

        st.subheader("📋 Dataset Information")
        st.write("### Shape")
        st.write(f"Rows : {rows}")
        st.write(f"Columns : {cols}")

        st.write("### Column Names")
        st.write(", ".join(map(str, data.columns)))

        st.write("### Data Types")
        render_table(data.dtypes.astype(str).rename("Type"), height=300)

        st.divider()

        st.subheader("📊 Summary Statistics")
        render_table(data.describe().round(2))

        st.divider()

        st.subheader("🔢 Mean")
        render_table(data.mean(numeric_only=True).rename("Mean"), height=300)

        st.subheader("📈 Median")
        render_table(data.median(numeric_only=True).rename("Median"), height=300)

        st.subheader("📉 Mode")
        render_table(data.mode().head(1))

        st.subheader("🔍 Unique Values")
        render_table(data.nunique().rename("Unique"), height=300)

    else:
        st.warning("⚠ Please upload a dataset first.")

elif page == "Visualizations":

    st.header("📈 Data Visualizations")
    st.markdown('<div class="page-description">Explore your data with interactive charts and smart chart recommendations.</div>', unsafe_allow_html=True)

    if "data" in st.session_state:

        data = st.session_state["data"]

        numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = data.select_dtypes(exclude=["number"]).columns.tolist()

        st.subheader("🧠 Smart Chart Recommendation")

        recommended_chart = ""
        reason = ""

        if len(numeric_columns) >= 2:
            recommended_chart = "Scatter Plot"
            reason = "Your dataset contains two or more numeric columns. Scatter Plot is ideal for identifying relationships between numeric variables."
        elif len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
            recommended_chart = "Bar Chart"
            reason = "Your dataset contains both categorical and numeric columns. Bar Chart is best for comparing categories."
        elif len(numeric_columns) == 1:
            recommended_chart = "Histogram"
            reason = "Only one numeric column was detected. Histogram is useful for understanding data distribution."
        elif len(categorical_columns) >= 1:
            recommended_chart = "Pie Chart"
            reason = "The dataset mainly contains categorical data. Pie Chart is useful for showing proportions."

        if recommended_chart:
            st.success(f"✅ Recommended Chart: {recommended_chart}")
            st.info(reason)

            if st.button("✨ Generate Recommended Chart"):

                if recommended_chart == "Scatter Plot":
                    fig = px.scatter(
                        data, x=numeric_columns[0], y=numeric_columns[1],
                        title=f"{numeric_columns[0]} vs {numeric_columns[1]}",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                elif recommended_chart == "Bar Chart":
                    fig = px.bar(
                        data, x=categorical_columns[0], y=numeric_columns[0],
                        title=f"{numeric_columns[0]} by {categorical_columns[0]}",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                elif recommended_chart == "Histogram":
                    fig = px.histogram(data, x=numeric_columns[0], title=f"Distribution of {numeric_columns[0]}")
                    st.plotly_chart(fig, use_container_width=True)

                elif recommended_chart == "Pie Chart":
                    value_counts = data[categorical_columns[0]].value_counts().reset_index()
                    value_counts.columns = [categorical_columns[0], "Count"]
                    fig = px.pie(
                        value_counts, names=categorical_columns[0], values="Count",
                        title=f"{categorical_columns[0]} Distribution",
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("📊 Manual Visualization")

        chart_type = st.selectbox(
            "Select Chart Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart"],
        )

        if chart_type == "Pie Chart":
            column = st.selectbox("Select Column", data.columns)
            value_counts = data[column].value_counts().reset_index()
            value_counts.columns = [column, "Count"]
            fig = px.pie(value_counts, names=column, values="Count", title=f"{column} Distribution")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Histogram":
            column = st.selectbox("Select Numeric Column", numeric_columns)
            fig = px.histogram(data, x=column, title=f"Histogram of {column}")
            st.plotly_chart(fig, use_container_width=True)

        else:
            x = st.selectbox("Select X-axis", data.columns)
            y = st.selectbox("Select Y-axis", numeric_columns)

            if chart_type == "Bar Chart":
                fig = px.bar(data, x=x, y=y)
            elif chart_type == "Line Chart":
                fig = px.line(data, x=x, y=y)
            elif chart_type == "Scatter Plot":
                fig = px.scatter(data, x=x, y=y)
            elif chart_type == "Box Plot":
                fig = px.box(data, x=x, y=y)

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠ Please upload a dataset first.")

elif page == "ML Lab":

    st.header("🤖 Machine Learning Lab")
    st.markdown(
        "Train machine learning models using your uploaded dataset."
    )

    if "data" not in st.session_state:
        st.warning("Please upload a dataset first.")

    else:
        data = st.session_state["data"].copy()

        st.subheader("Dataset Preview")
        render_table(data.head())

        task = st.selectbox(
            "Select ML Task",
            ["Regression", "Classification", "K-Means Clustering"]
        )

        if task == "K-Means Clustering":

            numeric_cols = data.select_dtypes(
                include="number"
            ).columns.tolist()

            if len(numeric_cols) < 2:
                st.warning(
                    "K-Means requires at least two numeric columns."
                )
            else:
                features = st.multiselect(
                    "Select Features",
                    numeric_cols,
                    default=numeric_cols[:2]
                )

                k = st.slider(
                    "Number of Clusters (K)",
                    min_value=2,
                    max_value=10,
                    value=3
                )

                if st.button("🚀 Run K-Means"):
                    if not features:
                        st.error("Select at least one feature.")
                    else:
                        try:
                            X = data[features].copy()
                            X = X.replace([float("inf"), float("-inf")], float("nan"))
                            X = SimpleImputer(strategy="mean").fit_transform(X)
                            X = StandardScaler().fit_transform(X)

                            if len(X) < k:
                                st.error("K cannot exceed the number of rows.")
                            else:
                                model = KMeans(
                                    n_clusters=k,
                                    random_state=42,
                                    n_init=10
                                )
                                labels = model.fit_predict(X)

                                result = data.copy()
                                result["Cluster"] = labels

                                st.success("Clustering completed!")
                                st.write("Cluster distribution:")
                                st.write(
                                    result["Cluster"].value_counts().sort_index()
                                )
                                render_table(result.head(20))

                                st.download_button(
                                    "⬇ Download Results",
                                    result.to_csv(index=False).encode("utf-8"),
                                    "cluster_results.csv",
                                    "text/csv"
                                )

                        except Exception as e:
                            st.error(f"Clustering failed: {e}")

        else:
            numeric_cols = data.select_dtypes(
                include="number"
            ).columns.tolist()

            target = st.selectbox(
                "Select Target Column",
                data.columns
            )

            features = st.multiselect(
                "Select Input Features",
                [c for c in numeric_cols if c != target]
            )

            test_size = st.slider(
                "Test Data Percentage",
                min_value=10,
                max_value=40,
                value=20
            )

            if task == "Regression":
                model_name = st.selectbox(
                    "Select Model",
                    ["Linear Regression", "Random Forest Regressor"]
                )
            else:
                model_name = st.selectbox(
                    "Select Model",
                    ["Logistic Regression", "Random Forest Classifier"]
                )

            if st.button("🚀 Train Model"):
                try:
                    if not features:
                        st.error("Select at least one input feature.")
                    elif target in features:
                        st.error("Target cannot be an input feature.")
                    elif data[target].isna().all():
                        st.error("Target column contains no valid values.")
                    else:
                        clean = data[features + [target]].copy()
                        clean = clean.replace(
                            [float("inf"), float("-inf")], float("nan")
                        )
                        clean = clean.dropna(subset=[target])

                        X = clean[features]
                        y = clean[target]

                        if task == "Regression":
                            if not pd.api.types.is_numeric_dtype(y):
                                st.error("Regression target must be numeric.")
                                st.stop()

                            model = (
                                LinearRegression()
                                if model_name == "Linear Regression"
                                else RandomForestRegressor(
                                    n_estimators=100,
                                    random_state=42
                                )
                            )

                        else:
                            if y.nunique() < 2:
                                st.error("Classification needs at least two classes.")
                                st.stop()

                            model = (
                                LogisticRegression(max_iter=1000)
                                if model_name == "Logistic Regression"
                                else RandomForestClassifier(
                                    n_estimators=100,
                                    random_state=42
                                )
                            )

                        if len(clean) < 5:
                            st.error("Upload at least 5 valid rows.")
                            st.stop()

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y,
                            test_size=test_size / 100,
                            random_state=42
                        )

                        pipeline = make_pipeline(
                            SimpleImputer(strategy="mean"),
                            StandardScaler(),
                            model
                        )

                        pipeline.fit(X_train, y_train)
                        predictions = pipeline.predict(X_test)

                        if task == "Regression":
                            st.metric(
                                "R² Score",
                                f"{r2_score(y_test, predictions):.3f}"
                            )
                            st.metric(
                                "Mean Absolute Error",
                                f"{mean_absolute_error(y_test, predictions):.3f}"
                            )
                        else:
                            st.metric(
                                "Accuracy",
                                f"{accuracy_score(y_test, predictions):.2%}"
                            )

                        results = X_test.copy()
                        results["Actual"] = y_test.values
                        results["Predicted"] = predictions

                        st.subheader("Predictions")
                        render_table(results.head(20))

                        st.download_button(
                            "⬇ Download Predictions",
                            results.to_csv(index=False).encode("utf-8"),
                            "ml_predictions.csv",
                            "text/csv"
                        )

                except Exception as e:
                    st.error(f"Model training failed: {e}")

elif page == "Upload History":



    st.header("🗂 Upload History")
    st.markdown('<div class="page-description">Review datasets previously uploaded to your InsightHub account.</div>', unsafe_allow_html=True)

    try:
        conn = connect_db()
        cursor = conn.cursor()

        if is_guest:
            cursor.execute("SELECT id, file_name, rows_count, columns_count, upload_date FROM upload_history WHERE user_id IS NULL ORDER BY upload_date DESC")
        else:
            cursor.execute(
                "SELECT id, file_name, rows_count, columns_count, upload_date FROM upload_history WHERE user_id = %s ORDER BY upload_date DESC",
                (user["id"],),
            )

        history = cursor.fetchall()
        conn.close()
    except MySQLError as e:
        history = []
        st.error(f"Could not reach the database: {e}")

    if history:
        df = pd.DataFrame(history, columns=["ID", "File Name", "Rows", "Columns", "Upload Date"])

        search = st.text_input("🔍 Search by File Name")
        if search:
            df = df[df["File Name"].str.contains(search, case=False)]

        render_table(df)
        st.metric("📊 Total Uploads", len(df))

        if st.button("🗑 Clear My Upload History"):
            conn = connect_db()
            if is_guest:
                cursor.execute("DELETE FROM upload_history WHERE user_id IS NULL")
            else:
                cursor.execute("DELETE FROM upload_history WHERE user_id = %s", (user["id"],))
            conn.commit()
            conn.close()
            st.success("✅ Upload history cleared successfully!")
            st.rerun()
    else:
        st.info("📂 No upload history found.")

elif page == "Reports":

    st.header("📄 Reports")
    st.markdown('<div class="page-description">Generate a shareable PDF summary of your current dataset.</div>', unsafe_allow_html=True)

    if "data" in st.session_state:

        data = st.session_state["data"]

        if st.button("📄 Generate PDF Report"):
            with st.spinner("Building your report..."):
                generate_pdf(data)

            st.success("✅ Report generated!")

            with open("InsightHub_Report.pdf", "rb") as pdf_file:
                st.download_button(
                    "⬇ Download PDF", pdf_file,
                    file_name="InsightHub_Report.pdf", mime="application/pdf",
                )

    else:
        st.warning("⚠ Please upload a dataset first.")

elif page == "About":

    st.header("ℹ️ About InsightHub")
    st.markdown('<div class="page-description">Learn about InsightHub, its capabilities, and the technology behind it.</div>', unsafe_allow_html=True)

    st.markdown("""
# 📊 InsightHub

InsightHub is a Smart Data Analytics Dashboard developed using Python and Streamlit.

## Features

✅ Secure Sign In / Sign Up

✅ Upload CSV and Excel files

✅ Data Cleaning

✅ Data Analysis

✅ Interactive Visualizations

✅ Per-user MySQL Upload History

✅ PDF Report Generation

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- MySQL
- bcrypt
- ReportLab

## Developed By

Krish Kumar 24BCD10057

BCA Data Science
""")

    st.markdown("---")

st.caption("© 2026 InsightHub | Developed using Streamlit & Python")