import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os


def load_naturalness_data(csv_path):
    """
    Load naturalness data from CSV file.
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")
    print(f"Datasets: {df['dataset'].unique()}")
    print(f"Databases: {df['db_id'].nunique()}")

    return df


def analyze_naturalness_distribution(df):
    """
    Analyze the distribution of naturalness classes (N1, N2, N3).
    """
    print("\n" + "=" * 60)
    print("NATURALNESS DISTRIBUTION ANALYSIS")
    print("=" * 60)

    # Table naturalness distribution (unique tables only)
    print("\n1. TABLE NATURALNESS DISTRIBUTION:")
    print("-" * 40)
    unique_tables = df.drop_duplicates(subset=["dataset", "db_id", "table_name"])
    table_naturalness = unique_tables["table_naturalness"].value_counts()
    table_percentages = (
        unique_tables["table_naturalness"].value_counts(normalize=True) * 100
    )

    for category in table_naturalness.index:
        count = table_naturalness[category]
        percent = table_percentages[category]
        print(f"{category}: {count} ({percent:.1f}%)")

    # Column naturalness distribution
    print("\n2. COLUMN NATURALNESS DISTRIBUTION:")
    print("-" * 40)
    column_naturalness = df["column_naturalness"].value_counts()
    column_percentages = df["column_naturalness"].value_counts(normalize=True) * 100

    for category in column_naturalness.index:
        count = column_naturalness[category]
        percent = column_percentages[category]
        print(f"{category}: {count} ({percent:.1f}%)")

    # Overall distribution (tables + columns combined)
    print("\n3. OVERALL NATURALNESS DISTRIBUTION:")
    print("-" * 40)
    all_naturalness = list(unique_tables["table_naturalness"]) + list(
        df["column_naturalness"]
    )
    overall_counter = Counter(all_naturalness)
    total_items = len(all_naturalness)

    for category in sorted(overall_counter.keys()):
        count = overall_counter[category]
        percent = (count / total_items) * 100
        print(f"{category}: {count} ({percent:.1f}%)")

    return table_naturalness, column_naturalness, overall_counter


def analyze_by_database(df):
    """
    Analyze naturalness distribution by database.
    """
    print("\n" + "=" * 60)
    print("NATURALNESS DISTRIBUTION BY DATABASE")
    print("=" * 60)

    for db_id in sorted(df["db_id"].unique()):
        db_data = df[df["db_id"] == db_id]
        print(f"\n{db_id.upper()}:")
        print("-" * len(db_id))

        # Table distribution for this database (unique tables only)
        unique_db_tables = db_data.drop_duplicates(subset=["table_name"])
        table_dist = unique_db_tables["table_naturalness"].value_counts()
        column_dist = db_data["column_naturalness"].value_counts()

        print(f"Tables: {dict(table_dist)}")
        print(f"Columns: {dict(column_dist)}")


def create_visualizations(df, output_dir):
    """
    Create visualizations for naturalness distribution.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set style and font
    plt.style.use("default")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14

    # Define custom colors for naturalness levels (better color palette)
    naturalness_colors = {
        "N1": "#4CAF50",  # Material Green for Regular (easiest)
        "N2": "#FF9800",  # Material Orange for Low (medium)
        "N3": "#F44336",  # Material Red for Least (hardest)
    }

    # Get unique tables for table statistics
    unique_tables = df.drop_duplicates(subset=["dataset", "db_id", "table_name"])

    # Get dataset name for titles
    # dataset_name = df["dataset"].iloc[0] if len(df) > 0 else "Unknown"

    # 1. Table vs Column Naturalness Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Table naturalness (unique tables only)
    table_counts = unique_tables["table_naturalness"].value_counts()

    # Create custom labels with count values only
    table_labels = []
    table_colors = []
    for label in table_counts.index:
        count = table_counts[label]
        percentage = (count / len(unique_tables)) * 100
        # Only show label if percentage is 5% or greater
        if percentage >= 8.0:
            table_labels.append(f"{percentage:.1f}% ({count})")
        else:
            table_labels.append("")
        table_colors.append(naturalness_colors.get(label, "#95a5a6"))  # Default gray

    wedges1, texts1 = ax1.pie(
        table_counts.values,
        labels=table_labels,
        startangle=90,
        colors=table_colors,
        textprops={"fontsize": 16, "fontfamily": "Times New Roman", "color": "black"},
    )
    # ax1.set_title("Table Naturalness Distribution")

    # Column naturalness
    column_counts = df["column_naturalness"].value_counts()

    # Create custom labels with count values only
    column_labels = []
    column_colors = []
    for label in column_counts.index:
        count = column_counts[label]
        percentage = (count / len(df)) * 100
        # Only show label if percentage is 5% or greater
        if percentage >= 8.0:
            column_labels.append(f"{percentage:.1f}% ({count})")
        else:
            column_labels.append("")
        column_colors.append(naturalness_colors.get(label, "#95a5a6"))  # Default gray

    wedges2, texts2 = ax2.pie(
        column_counts.values,
        labels=column_labels,
        startangle=90,
        colors=column_colors,
        textprops={"fontsize": 16, "fontfamily": "Times New Roman", "color": "black"},
    )
    # ax2.set_title("Column Naturalness Distribution")

    # Create single legend for both pie charts
    all_categories = sorted(set(table_counts.index) | set(column_counts.index))
    legend_labels = []
    legend_colors = []
    for label in all_categories:
        if label == "N1":
            legend_labels.append("N1: Regular (e.g., AcquisitionTaxAccount)")
        elif label == "N2":
            legend_labels.append("N2: Low (e.g., AcqAcct)")
        elif label == "N3":
            legend_labels.append("N3: Least (e.g., AcAct)")
        else:
            legend_labels.append(label)
        legend_colors.append(naturalness_colors.get(label, "#95a5a6"))

    # Create custom legend handles
    from matplotlib.patches import Patch

    legend_handles = [Patch(color=color) for color in legend_colors]

    # Place single legend at the bottom center with more spacing and larger size
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(legend_labels),
        fontsize=20,
        frameon=True,
        fancybox=True,
        shadow=True,
        prop={"family": "Times New Roman", "size": 20},
        markerscale=2.0,
        borderpad=1.2,
        labelspacing=1.0,
        handlelength=2.5,
        handletextpad=1.0,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Increased bottom margin for larger legend
    plt.savefig(
        f"{output_dir}/naturalness_distribution_pie.pdf", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 2. Bar chart comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for comparison
    categories = sorted(
        set(unique_tables["table_naturalness"].unique())
        | set(df["column_naturalness"].unique())
    )
    table_data = [table_counts.get(cat, 0) for cat in categories]
    column_data = [column_counts.get(cat, 0) for cat in categories]

    x = range(len(categories))
    width = 0.35

    # Create bars with same colors but different patterns/transparency
    bar_colors = [naturalness_colors.get(cat, "#95a5a6") for cat in categories]

    bars1 = ax.bar(
        [i - width / 2 for i in x],
        table_data,
        width,
        label="Tables",
        alpha=0.8,
        color=bar_colors,
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        column_data,
        width,
        label="Columns",
        alpha=0.5,
        color=bar_colors,
        edgecolor="black",
        linewidth=1.5,
        hatch="///",
    )

    # Add percentage labels on bars
    total_tables = len(unique_tables)
    total_columns = len(df)

    for bar, count in zip(bars1, table_data):
        height = bar.get_height()
        if height > 0:
            percentage = (count / total_tables) * 100
            if percentage >= 8.0:  # Only show label if percentage is 5% or greater
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.5,
                    f"{percentage:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=16,
                    fontfamily="Times New Roman",
                    color="black",
                )

    for bar, count in zip(bars2, column_data):
        height = bar.get_height()
        if height > 0:
            percentage = (count / total_columns) * 100
            if percentage >= 8.0:  # Only show label if percentage is 5% or greater
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.5,
                    f"{percentage:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=16,
                    fontfamily="Times New Roman",
                    color="black",
                )

    ax.set_xlabel("Naturalness Categories", fontsize=20, fontfamily="Times New Roman")
    ax.set_ylabel("Count", fontsize=20, fontfamily="Times New Roman")
    # ax.set_title(f"Table vs Column Naturalness Distribution of {dataset_name}")
    ax.set_xticks(x)

    # Create custom tick labels with descriptions
    tick_labels = []
    for cat in categories:
        if cat == "N1":
            tick_labels.append("N1: Regular\n(e.g., AcquisitionTaxAccount)")
        elif cat == "N2":
            tick_labels.append("N2: Low\n(e.g., AcqAcct)")
        elif cat == "N3":
            tick_labels.append("N3: Least\n(e.g., AcAct)")
        else:
            tick_labels.append(cat)

    ax.set_xticklabels(tick_labels, fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(
        fontsize=20,
        frameon=True,
        fancybox=True,
        shadow=True,
        prop={"family": "Times New Roman", "size": 20},
        markerscale=1.5,
        borderpad=1.2,
        labelspacing=0.8,
        handlelength=2.5,
        handletextpad=1.0,
    )
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/naturalness_distribution_bar.pdf", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 3. Distribution by database
    fig, ax = plt.subplots(figsize=(16, 9))

    # Create stacked bar chart
    db_ids = sorted(df["db_id"].unique())
    bottom = [0] * len(db_ids)

    for cat in categories:
        values = []
        for db_id in db_ids:
            db_data = df[df["db_id"] == db_id]
            unique_db_tables = db_data.drop_duplicates(subset=["table_name"])
            all_naturalness = list(unique_db_tables["table_naturalness"]) + list(
                db_data["column_naturalness"]
            )
            counter = Counter(all_naturalness)
            values.append(counter.get(cat, 0))

        # Create custom label for legend
        if cat == "N1":
            legend_label = "N1: Regular (e.g., AcquisitionTaxAccount)"
        elif cat == "N2":
            legend_label = "N2: Low (e.g., AcqAcct)"
        elif cat == "N3":
            legend_label = "N3: Least (e.g., AcAct)"
        else:
            legend_label = cat

        bars = ax.bar(
            db_ids,
            values,
            bottom=bottom,
            label=legend_label,
            alpha=0.85,
            color=naturalness_colors.get(cat, "#95a5a6"),
            edgecolor="white",
            linewidth=0.5,
        )

        # Add percentage labels on each section of the stacked bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            if value > 0:
                # Calculate total items for this database
                db_data = df[df["db_id"] == db_ids[i]]
                unique_db_tables = db_data.drop_duplicates(subset=["table_name"])
                total_items = len(unique_db_tables) + len(db_data)
                percentage = (value / total_items) * 100

                if percentage >= 8.0:  # Only show label if percentage is 8% or greater
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bottom[i] + value / 2,
                        f"{percentage:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=18,
                        fontfamily="Times New Roman",
                        color="black",  # Remove bold styling
                    )

        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xlabel("Database ID", fontsize=22, fontfamily="Times New Roman")
    ax.set_ylabel("Table & Column Count", fontsize=22, fontfamily="Times New Roman")
    ax.legend(
        fontsize=20,
        frameon=True,
        fancybox=True,
        shadow=True,
        prop={"family": "Times New Roman", "size": 20},
        loc="upper right",
        markerscale=1.5,
        borderpad=1.2,
        labelspacing=0.8,
        handlelength=2.5,
        handletextpad=1.0,
    )
    plt.xticks(rotation=45, ha="right", fontsize=18)
    ax.tick_params(axis="y", labelsize=18)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/naturalness_by_database.pdf", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"\nVisualization plots saved to: {output_dir}")


def generate_summary_report(df, output_dir):
    """
    Generate a summary report of the naturalness analysis.
    """
    output_path = output_dir + "/naturalness_summary.txt"
    unique_tables = df.drop_duplicates(subset=["dataset", "db_id", "table_name"])

    with open(output_path, "w") as f:
        f.write("DATABASE NATURALNESS ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total Records: {len(df)}\n")
        f.write(f"Total Databases: {df['db_id'].nunique()}\n")
        f.write(f"Total Unique Tables: {len(unique_tables)}\n")
        f.write(f"Total Columns: {len(df)}\n\n")

        f.write("TABLE NATURALNESS DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        table_dist = unique_tables["table_naturalness"].value_counts()
        for cat, count in table_dist.items():
            percent = (count / len(unique_tables)) * 100
            f.write(f"{cat}: {count} ({percent:.1f}%)\n")

        f.write("\nCOLUMN NATURALNESS DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        column_dist = df["column_naturalness"].value_counts()
        for cat, count in column_dist.items():
            percent = (count / len(df)) * 100
            f.write(f"{cat}: {count} ({percent:.1f}%)\n")

    print(f"Summary report saved to: {output_path}")


if __name__ == "__main__":
    # Configuration
    csv_path = "data/augmentation/decrease_naturalness/databases_naturalness.csv"
    output_dir = "data/augmentation/decrease_naturalness/databases_naturalness_analysis"

    df = load_naturalness_data(csv_path)

    if df is not None:
        table_dist, column_dist, overall_dist = analyze_naturalness_distribution(df)
        analyze_by_database(df)
        create_visualizations(df, output_dir)
        generate_summary_report(df, output_dir)
