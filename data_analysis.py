"""A module for analyzing election and city connection data. Most importantly, it combines the datasets for election and city connections."""
import json
from pathlib import Path
import db.static as static

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import linregress
from ydata_profiling import ProfileReport
import numpy as np
from scipy.stats import shapiro, normaltest




import dataframe_image as dfi
from pathlib import Path

import sys  
sys.stdout.reconfigure(encoding="utf-8")



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np


def distance_vote_linear_models(csv_filename: str = "db/combined_data.csv") -> None:
    """
    Pro čtyři vybrané strany (Piráti, SPD, SPOLU, ANO) natrénuje lineární regresi
    mezi vzdáleností od nejbližšího krajského města a procentuálním výsledkem strany.
    Použije train/test split, vypíše metriky a vykreslí scatter s fitovanou přímkou,
    zvlášť barevně pro trénovací a testovací část.
    """
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / csv_filename

    df = pd.read_csv(csv_path, encoding="utf-8")

    # Vzdálenost k nejbližšímu krajskému městu (definice přes krajská města)
    df["distance"] = df[["Praha", "Brno"]].min(axis=1)
    df = df[df["distance"] > 0]

    # Mapování stran na sloupce s procenty v datech
    party_columns = {
        "Piráti": "P16_PROC",
        "SPD": "P6_PROC",
        "SPOLU": "P11_PROC",
        "ANO": "P22_PROC",
    }

    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for party_name, col in party_columns.items():
        if col not in df.columns:
            print(f"Sloupec {col} pro stranu {party_name} v datech chybí, přeskočeno.")
            continue

        party_df = df[["distance", col]].dropna()
        if party_df.empty:
            print(f"Strana {party_name}: žádná dostupná data, přeskočeno.")
            continue

        X = party_df[["distance"]].values
        y = party_df[col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        r2_test = r2_score(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        mae = mean_absolute_error(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)  # <- fix: no 'squared' keyword

        print(f"\n=== {party_name} ({col}) ===")
        print(f"slope (β1):          {model.coef_[0]:.4f} p. b. / minuta")
        print(f"intercept (β0):      {model.intercept_: .4f}")
        print(f"R² (train):          {r2_train:.4f}")
        print(f"R² (test):           {r2_test:.4f}")
        print(f"MAE (test):          {mae:.4f}")
        print(f"RMSE (test):         {rmse:.4f}")

        # Data pro vizualizaci (train/test odlišíme barvou)
        plot_train = pd.DataFrame(
            {"distance": X_train.ravel(), "vote_share": y_train, "set": "Train"}
        )
        plot_test = pd.DataFrame(
            {"distance": X_test.ravel(), "vote_share": y_test, "set": "Test"}
        )
        plot_df = pd.concat([plot_train, plot_test], ignore_index=True)

        # Přímka predikce
        x_range = np.linspace(
            party_df["distance"].min(), party_df["distance"].max(), 200
        ).reshape(-1, 1)
        y_range = model.predict(x_range)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=plot_df,
            x="distance",
            y="vote_share",
            hue="set",
            palette={"Train": "tab:blue", "Test": "tab:orange"},
            alpha=0.8,
        )

        plt.plot(
            x_range.ravel(),
            y_range,
            color="black",
            linewidth=2,
            label="Fitted line",
        )

        plt.xlabel("Vzdálenost k nejbližšímu krajskému městu (minuty)")
        plt.ylabel("Volební výsledek strany (v %)")
        plt.title(f"Lineární regrese: {party_name} vs. vzdálenost")
        plt.legend()
        plt.tight_layout()

        out_path = plots_dir / f"linreg_{party_name.lower()}_vs_distance.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.show()


def p6_vs_distance_scatter(csv_filename: str = "db/combined_data.csv") -> None:
    """
    Vytvoří scatterplot vztahu mezi volebním výsledkem strany P6 (v procentech)
    a vzdáleností od nejbližšího krajského města.
    """
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / csv_filename

    # Načtení dat
    df = pd.read_csv(csv_path, encoding="utf-8")

    # Vzdálenost k nejbližšímu krajskému městu
    df["distance"] = df[["Praha", "Brno"]].min(axis=1)

    # Filtrování: jen kladné vzdálenosti a nechybející P6_PROC
    df = df[(df["distance"] > 0)].dropna(subset=["P6_PROC"])

    # Scatterplot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="distance",
        y="P23_PROC",
        alpha=0.7,
    )

    plt.xlabel("Vzdálenost k nejbližšímu krajskému městu (minuty)")
    plt.ylabel("Volební výsledek P6 (v %)")
    plt.title("Vztah mezi výsledkem P6 a vzdáleností od nejbližšího krajského města")
    plt.tight_layout()

    # Uložení grafu
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "p6_vs_distance_scatter.png",
                dpi=300, bbox_inches="tight")
    plt.show()


def population_vs_distance_scatter(csv_filename: str = "db/combined_data.csv") -> None:
    """
    Vytvoří scatterplot vztahu mezi velikostí populace města a vzdáleností
    od nejbližšího krajského města.
    """
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / csv_filename

    # Načtení dat
    df = pd.read_csv(csv_path, encoding="utf-8")

    # Drop rows with regional cities
    df = df[~df["city_name"].isin(static.REGIONAL_CITIES_NAMES)]
    # Vzdálenost k nejbližšímu krajskému městu
    df["distance"] = df[static.REGIONAL_CITIES_NAMES].min(axis=1)
    df = df[df["distance"] > 0].dropna(subset=["population", "distance"])

    # Scatterplot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="distance",
        y="population",
        alpha=0.7,
    )

    plt.xlabel("Vzdálenost k nejbližšímu krajskému městu (minuty)")
    plt.ylabel("Počet obyvatel")
    plt.title("Vztah mezi velikostí města a vzdáleností od nejbližšího krajského města")
    plt.tight_layout()

    # Uložení grafu
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "population_vs_distance_scatter.png",
                dpi=300, bbox_inches="tight")
    plt.show()


def top_10_most_distant_city_center_pairs(csv_filename: str = "db/combined_data.csv") -> None:
    """
    Najde 10 nejvzdálenějších dvojic (město - krajské město) a vizualizuje je
    podobně jako swarmplot, ale místo teček vykreslí názvy měst.
    Barva textu odpovídá vzdálenosti (gradient).
    """
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / csv_filename

    # Načtení dat
    df = pd.read_csv(csv_path, encoding="utf-8")

    # Široký -> dlouhý formát: sloupce krajských měst
    long_df = df.melt(
        id_vars=["city_name"],
        value_vars=static.REGIONAL_CITIES_NAMES,
        var_name="regional_city",
        value_name="distance",
    )

    # Odstranit chybějící a nulové vzdálenosti
    long_df = long_df.dropna(subset=["distance"])
    long_df = long_df[long_df["distance"] > 0]

    # 10 nejvzdálenějších dvojic
    top10 = long_df.nlargest(10, "distance").copy()
    top10 = top10.sort_values("distance", ascending=True)

    # Mapování krajských měst na "základní" y-pozice
    unique_regions = list(top10["regional_city"].unique())
    region_to_y = {region: i for i, region in enumerate(unique_regions)}
    top10["y_base"] = top10["regional_city"].map(region_to_y)

    # Větší vertikální jitter pro lepší rozestupy textů – správně přiřazený podle indexu
    top10["y"] = np.nan
    for region, group in top10.groupby("regional_city"):
        n = len(group)
        if n == 1:
            offsets = np.array([0.0])
        else:
            offsets = np.linspace(-0.35, 0.35, n)
        top10.loc[group.index, "y"] = group["y_base"].values + offsets

    # Příprava colormapy pro barvení textu podle vzdálenosti
    dist_min = top10["distance"].min()
    dist_max = top10["distance"].max()
    norm = plt.Normalize(dist_min, dist_max)
    cmap = plt.cm.viridis

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))

    # Jemná svislá mřížka
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.grid(axis="y", linestyle="")

    # Oddělující horizontální čáry pro jednotlivá krajská města
    for y in region_to_y.values():
        ax.axhline(
            y=y,
            color="lightgray",
            linewidth=0.6,
            alpha=0.8,
            zorder=0,
        )

    # Text místo bodů, s barvou dle vzdálenosti
    for _, row in top10.iterrows():
        color = cmap(norm(row["distance"]))
        ax.text(
            row["distance"],
            row["y"],
            row["city_name"],
            va="center",
            ha="left",
            color=color,
            fontsize=10,
        )

    # Nastavení os
    ax.set_xlabel("Vzdálenost (minuty)")
    ax.set_ylabel("Krajské město")
    ax.set_title("10 nejvzdálenějších dvojic město – krajské město (gradientový textový swarm)")
    ax.set_yticks(list(region_to_y.values()))
    ax.set_yticklabels(list(region_to_y.keys()))
    ax.set_ylim(-0.7, len(unique_regions) - 0.3)
    ax.set_xlim(0, top10["distance"].max() * 1.05)

    # Colorbar pro legendu gradientu
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Vzdálenost (minuty)")

    sns.despine(left=False, bottom=False)
    plt.tight_layout()

    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        plots_dir / "top10_distant_city_center_pairs_text_swarm_gradient.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()



def distance_to_regional_city_histogram(
    csv_filename: str = "db/combined_data.csv",
) -> None:
    """
    Plot a histogram of distances to regional cities using seaborn and matplotlib.

    Parameters
    ----------
    csv_filename : str
        Name of the CSV file with the combined data (must contain a 'distance' column).
        By default, assumes 'db/combined_data.csv' in the same directory as this script.
    """
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / csv_filename

    # Load data
    df = pd.read_csv(csv_path, encoding="utf-8")
    # min of these cols: Praha,Brno,Ostrava,Plzeň,Liberec,Olomouc,Ústí nad Labem,Hradec Králové,Pardubice,Zlín,České Budějovice,Jihlava,Karlovy Vary
    df["distance"] = df[static.REGIONAL_CITIES_NAMES].min(axis=1)
    # remove entries with distance == 0
    df = df[df["distance"] > 0]

    # Create histogram
    sns.histplot(df["distance"], bins=10, kde=False)

    plt.xlabel("Vzdálenost k nejbližšímu krajskému městu (minuty)")
    plt.ylabel("Počet měst")
    plt.title("Distribuce vzdáleností k nejbližšímu krajskému městu")
    plt.tight_layout()
    # let the x axis have ticks by 10
    plt.xticks(np.arange(0, df["distance"].max() + 10, 10))
    # plt.show()
    # save to file
    plt.savefig("plots/distances_histogram.png")

    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
 
    closest_cities = df.nsmallest(5, "distance")[["city_name", "distance"]].copy()
    farthest_cities = df.nlargest(5, "distance")[["city_name", "distance"]].copy()

    closest_cities["distance"] = closest_cities["distance"].round().astype(int)
    farthest_cities["distance"] = farthest_cities["distance"].round().astype(int)

    dfi.export(closest_cities.style.hide(axis="index"), plots_dir / "closest_cities_table.png")
    dfi.export(farthest_cities.style.hide(axis="index"), plots_dir / "farthest_cities_table.png")
        





def distance_to_centers(
    csv_filename: str = "db/combined_data.csv",
) -> None:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / csv_filename

    # Load data
    df = pd.read_csv(csv_path, encoding="utf-8")
    df["distance"] = df[["Praha", "Brno"]].min(axis=1)
    # remove entries with distance == 0
    df = df[df["distance"] > 0]
    distances = df["distance"]

    sh_stat, sh_p = normaltest(distances)
    print("Shapiro-Wilk test on distance")
    print(f"  W statistic: {sh_stat:.4f}")
    print(f"  p-value:     {sh_p:.4f}")

    alpha = 0.05
    if sh_p < alpha:
        print(f"\n At a = {alpha}, we reject normality of distance.")
    else:
        print(f"\n At a = {alpha}, we do not reject normality of distance.")
    # Create histogram
    sns.histplot(df["distance"], bins=10, kde=False)

    plt.xlabel("Vzdálenost k nejbližšímu centru (minuty)")
    plt.ylabel("Počet měst")
    plt.title("Distribuce vzdáleností k Praze nebo Brnu")
    plt.tight_layout()
    # let the x axis have ticks by 20
    plt.xticks(np.arange(0, df["distance"].max() + 20, 20))
    plt.show()
    # save to file
    #plt.savefig("plots/distances_histogram_centers.png")

    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
 
    closest_cities = df.nsmallest(5, "distance")[["city_name", "distance"]].copy()
    farthest_cities = df.nlargest(5, "distance")[["city_name", "distance"]].copy()

    closest_cities["distance"] = closest_cities["distance"].round().astype(int)
    farthest_cities["distance"] = farthest_cities["distance"].round().astype(int)

    dfi.export(closest_cities.style.hide(axis="index"), plots_dir / "closest_cities_table_2.png")
    dfi.export(farthest_cities.style.hide(axis="index"), plots_dir / "farthest_cities_table_2.png")
        


def average_distance_ranking2(csv_filename: str = "db/combined_data.csv") -> None:
    """
    Pro každé krajské město spočítá průměrnou vzdálenost všech měst
    k tomuto centru a výsledek zobrazí pomocí horizontálního barplotu.
    """
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / csv_filename

    df = pd.read_csv(csv_path, encoding="utf-8")

    # Průměrná vzdálenost pro každé krajské město (sloupec)
    avg_distances = (
        df[static.REGIONAL_CITIES_NAMES]
        .mean(axis=0, skipna=True)
        .to_frame(name="average_distance")
        .rename_axis("regional_city")
        .reset_index()
    )

    # Seřadit podle průměrné vzdálenosti (vzestupně = "nejdostupnější" první)
    avg_distances = avg_distances.sort_values("average_distance", ascending=True)

    # Vizualizace
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=avg_distances,
        x="average_distance",
        y="regional_city",
        orient="h",
    )

    plt.xlabel("Průměrná vzdálenost všech měst (minuty)")
    plt.ylabel("Krajské město")
    plt.title("Průměrná vzdálenost měst k jednotlivým krajským centrům")
    plt.tight_layout()

    # Uložení grafu
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "average_distance_ranking.png", dpi=300, bbox_inches="tight")
    plt.show()



def average_distance_ranking(csv_filename: str = "db/combined_data.csv") -> None:
    """
    Pro každé krajské město spočítá průměrnou vzdálenost všech měst
    k tomuto centru a výsledek zobrazí pomocí lollipop grafu.
    """
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / csv_filename

    df = pd.read_csv(csv_path, encoding="utf-8")

    # Průměrná vzdálenost pro každé krajské město
    avg_distances = (
        df[static.REGIONAL_CITIES_NAMES]
        .mean(axis=0, skipna=True)
        .to_frame(name="average_distance")
        .rename_axis("regional_city")
        .reset_index()
    )

    # Seřadit podle průměrné vzdálenosti
    avg_distances = avg_distances.sort_values("average_distance", ascending=True)

    # Lollipop graf: osa x = průměrná vzdálenost, osa y = město
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # vodorovné čáry (stonek)
    for _, row in avg_distances.iterrows():
        ax.hlines(
            y=row["regional_city"],
            xmin=0,
            xmax=row["average_distance"],
            linewidth=2,
            alpha=0.6,
        )

    # body (hlavička)
    sns.scatterplot(
        data=avg_distances,
        x="average_distance",
        y="regional_city",
        s=120,
        ax=ax,
    )

    plt.xlabel("Průměrná vzdálenost k městům (minuty)")
    plt.ylabel("Krajské město")
    plt.title("Průměrná vzdálenost krajských měst k městům v minutách")
    plt.tight_layout()

    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "average_distance_ranking_lollipop.png")
    plt.show()
    
if __name__ == "__main__":
    distance_vote_linear_models()
    # election_percentage_histogram()
    #distance_to_centers()
    #average_distance_ranking()
    # analyze_party25_6_vs_distance()
    


    
