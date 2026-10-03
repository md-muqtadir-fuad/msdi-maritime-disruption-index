"""Shanghai against the lead container port of every major trading country.

Purpose: show why a Shanghai-origin freight index carries information beyond Shanghai.
Each country's largest container port is placed on a world map, sized by container port
calls, coloured by how strongly its monthly activity co-moves with Shanghai's, and linked
to Shanghai by bilateral liner-shipping connectivity.

Exploratory. Port calls are an activity measure, not a freight-rate measure, so this
figure motivates SCFI's reach rather than testing it.

Sources
    IMF PortWatch daily port data  data/01_raw/controls/portwatch
    IMF PortWatch ports database (coordinates)  portwatch_ports_coordinates.csv
    Natural Earth 110m land (public domain)  data/01_raw/geo/ne_110m_land.geojson

Outputs
    outputs/figures/figure24_shanghai_vs_other_ports.{pdf,png}
    outputs/reports/shanghai_vs_other_ports.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402
import matplotlib.patheffects as path_effects  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PORTWATCH = PROJECT_ROOT / "data/01_raw/controls/portwatch/Daily_Ports_Data.csv"
LAND_GEOJSON = PROJECT_ROOT / "data/01_raw/geo/ne_110m_land.geojson"
LSBCI = PROJECT_ROOT / "data/01_raw/controls/connectivity/US_LSBCI.csv"
PORT_COORDS_CSV = (PROJECT_ROOT /
                   "data/01_raw/controls/portwatch/portwatch_ports_coordinates.csv")
FIGURE_DIR = PROJECT_ROOT / "outputs/figures"
REPORT_DIR = PROJECT_ROOT / "outputs/reports"

FOCUS_PORT = "Shanghai"
LON_CUT = -60.0  # map runs LON_CUT .. LON_CUT+360, Pacific-centred
N_COUNTRIES = 30
MIN_MONTHS = 48

# Some PortWatch country strings carry encoding artefacts that will never match the
# LSBCI label (Turkiye is stored mis-encoded), so key those by ISO3 code instead.
LSBCI_BY_ISO3: dict[str, str] = {
    "TUR": "Turkiye",
}

# PortWatch country label -> UNCTAD LSBCI partner label. Explicit because the two
# sources name economies differently and a silent mismatch would drop a lane.
LSBCI_PARTNER: dict[str, str] = {
    "Hong Kong SAR": "China, Hong Kong SAR",
    "Korea": "Republic of Korea",
    "Taiwan Province of China": "China, Taiwan Province of",
    "The Netherlands": "Netherlands (Kingdom of the)",
    "Vietnam": "Viet Nam",
    "United States": "United States",
    "Singapore": "Singapore",
    "Malaysia": "Malaysia",
    "Japan": "Japan",
    "Belgium": "Belgium",
    "United Arab Emirates": "United Arab Emirates",
    "Indonesia": "Indonesia",
    "Thailand": "Thailand",
    "Germany": "Germany",
    "Sri Lanka": "Sri Lanka",
    "Morocco": "Morocco",
    "Spain": "Spain",
    "Greece": "Greece",
    "Philippines": "Philippines",
    "Colombia": "Colombia",
    "India": "India",
    "Brazil": "Brazil",
    "Saudi Arabia": "Saudi Arabia",
    "Panama": "Panama",
    "France": "France",
    "Italy": "Italy",
    "Jamaica": "Jamaica",
    "Malta": "Malta",
}

# Project palette (as used by stages 24 and 48).
PUBLICATION_COLORS = {
    "primary": "#1f4e79",
    "secondary": "#c55a11",
    "robustness": "#6b7280",
    "warning": "#b91c1c",
}
COLORS = {
    "land": "#e9e9e7",
    "land_edge": "#c4c4c0",
    "ocean": "#f7fafc",
    "focus": "#f2b705",
    "focus_label": "#111111",
    "link_high": PUBLICATION_COLORS["warning"],
    "link_low": "#111111",
    "trend": PUBLICATION_COLORS["robustness"],
}

# No stage-24 figure uses a colormap, so build the sequential ramp from the project's
# primary colour rather than importing an off-palette scheme.
HOUSE_CMAP = LinearSegmentedColormap.from_list(
    "house_sequential", ["#dbe5ee", "#7fa0bd", PUBLICATION_COLORS["primary"]]
)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def load_port_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly container calls by port, plus the lead port of each country."""
    if not PORTWATCH.is_file():
        raise FileNotFoundError(f"PortWatch extract not found: {PORTWATCH}")
    frame = pd.read_csv(
        PORTWATCH,
        usecols=["date", "portname", "country", "ISO3", "portid",
                 "portcalls_container"],
        low_memory=False,
    )
    frame["date"] = pd.to_datetime(frame["date"], format="%Y/%m/%d", errors="coerce")
    frame = frame.dropna(subset=["date"])

    totals = frame.groupby(
        ["country", "ISO3", "portname", "portid"], as_index=False
    )["portcalls_container"].sum()
    lead = (
        totals.sort_values("portcalls_container", ascending=False)
        .groupby("country", as_index=False)
        .first()
        .sort_values("portcalls_container", ascending=False)
    )
    lead = lead[lead["portcalls_container"] > 0].head(N_COUNTRIES)

    wanted = set(lead["portname"]) | {FOCUS_PORT}
    monthly = (
        frame[frame["portname"].isin(wanted)]
        .groupby([pd.Grouper(key="date", freq="ME"), "portname"])[
            "portcalls_container"
        ]
        .sum()
        .unstack("portname")
    )
    last_day = frame["date"].max()
    if last_day.day < 28:  # source ends mid-month
        monthly = monthly.iloc[:-1]
    return monthly, lead


def comovement(monthly: pd.DataFrame) -> pd.Series:
    changes = np.log(monthly.replace(0, np.nan)).diff()
    usable = changes.columns[changes.notna().sum() >= MIN_MONTHS]
    corr = changes[usable].corr()[FOCUS_PORT].drop(FOCUS_PORT, errors="ignore")
    return corr.dropna().sort_values(ascending=False)


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in kilometres."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def shift_lon(lon: float) -> float:
    """Longitude on the Pacific-centred axis."""
    return lon + 360.0 if lon < LON_CUT else lon


def draw_land(ax: plt.Axes) -> None:
    """Land polygons, drawn twice (at lon and lon+360) so the wrap is seamless."""
    geo = json.loads(LAND_GEOJSON.read_text(encoding="utf-8"))
    for feature in geo.get("features", []):
        geom = feature.get("geometry") or {}
        kind, coords = geom.get("type"), geom.get("coordinates")
        rings = []
        if kind == "Polygon":
            rings = [coords[0]]
        elif kind == "MultiPolygon":
            rings = [part[0] for part in coords]
        for ring in rings:
            pts = np.asarray(ring, dtype=float)
            for offset in (0.0, 360.0):
                shifted = pts.copy()
                shifted[:, 0] += offset
                ax.add_patch(
                    Polygon(
                        shifted,
                        closed=True,
                        facecolor=COLORS["land"],
                        edgecolor=COLORS["land_edge"],
                        linewidth=0.35,
                        zorder=1,
                    )
                )


def load_coordinates() -> pd.DataFrame:
    """Port positions from the IMF PortWatch port database, joined on portid."""
    if not PORT_COORDS_CSV.is_file():
        raise FileNotFoundError(
            f"Port coordinates not found: {PORT_COORDS_CSV}. Fetch them from the "
            "IMF PortWatch ports FeatureServer before running this stage."
        )
    frame = pd.read_csv(PORT_COORDS_CSV)
    return frame.set_index("portid")[["lat", "lon"]]


def load_lsbci() -> dict[str, float]:
    """Latest bilateral liner-shipping connectivity between China and each partner.

    Used as the flow proxy on the map. PortWatch has no origin-destination volumes, so
    the strength of a Shanghai-to-port link is represented by how well connected China
    and that economy are by container liner services.
    """
    if not LSBCI.is_file():
        raise FileNotFoundError(f"LSBCI extract not found: {LSBCI}")
    frame = pd.read_csv(LSBCI, low_memory=False)
    frame.columns = [c.strip() for c in frame.columns]
    china = frame[frame["Economy Label"].astype(str).str.strip().eq("China")]
    china = china.dropna(subset=["Index"])
    latest = china[china["Quarter"].eq(china["Quarter"].max())]
    return dict(zip(latest["Partner Label"].astype(str).str.strip(), latest["Index"]))


def place_labels(fig, ax, ports: pd.DataFrame, fontsize: float = 6.4) -> None:
    """Greedy first-fit label placement in display space.

    A world map puts several ports within a few pixels of each other (the European
    range, the Pearl River delta), so fixed offsets collide. Each label tries a ring
    of candidate positions and takes the first that clears everything already placed.
    """
    fig.canvas.draw()
    candidates = [
        (0, -9.5), (0, 7.5), (12, -2), (-12, -2),
        (10, -9), (-10, -9), (10, 7), (-10, 7),
        (0, -17), (0, 15), (19, -2), (-19, -2),
        (17, -13), (-17, -13), (17, 11), (-17, 11),
        (0, -25), (0, 23), (26, -2), (-26, -2),
        (24, -18), (-24, -18), (24, 16), (-24, 16),
    ]
    placed: list[tuple[float, float, float, float]] = []
    # transData works in pixels while annotate offsets are in points
    scale = fig.dpi / 72.0
    order = ports.sort_values("portcalls_container", ascending=False)
    for row in order.itertuples():
        px, py = ax.transData.transform((row.x, row.lat))
        w = (3.9 * len(row.ISO3) + 4.0) * scale
        h = (fontsize + 4.0) * scale
        chosen = candidates[0]
        for dx, dy in candidates:
            x0, y0 = px + dx * scale - w / 2, py + dy * scale - h / 2
            if not any(x0 < bx + bw and x0 + w > bx and y0 < by + bh and y0 + h > by
                       for bx, by, bw, bh in placed):
                chosen = (dx, dy)
                placed.append((x0, y0, w, h))
                break
        else:
            placed.append((px + chosen[0] * scale - w / 2,
                           py + chosen[1] * scale - h / 2, w, h))
        ax.annotate(
            row.ISO3, xy=(row.x, row.lat), xytext=chosen,
            textcoords="offset points", ha="center", va="center",
            fontsize=fontsize, color="#3a3a3a", zorder=7,
            path_effects=[path_effects.withStroke(linewidth=1.8,
                                                  foreground="white")],
        )


def build_figure(table: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(12.2, 4.6), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(COLORS["ocean"])
    draw_land(ax)

    mapped = table.dropna(subset=["lat", "lon"]).copy()
    mapped["x"] = mapped["lon"].map(shift_lon)
    focus = mapped[mapped["portname"].eq(FOCUS_PORT)]
    others = mapped[~mapped["portname"].eq(FOCUS_PORT)]

    # curved connectors, coloured by bilateral liner connectivity
    if not focus.empty:
        ox, oy = float(focus["x"].iloc[0]), float(focus["lat"].iloc[0])
        flow = others["connectivity"]
        if flow.notna().any():
            cut = float(flow.median())
            for row in others.itertuples():
                if pd.isna(row.connectivity):
                    continue
                strong = row.connectivity >= cut
                # ports wrapped past the cut sit east of Shanghai on this axis; bow
                # their arcs the other way so they clear the Pacific
                rad = -0.16 if row.lon < LON_CUT else 0.16
                ax.add_patch(
                    FancyArrowPatch(
                        (ox, oy), (row.x, row.lat),
                        connectionstyle=f"arc3,rad={rad}",
                        arrowstyle="-",
                        linewidth=0.75,
                        color=COLORS["link_high"] if strong else COLORS["link_low"],
                        alpha=0.92 if strong else 0.78,
                        zorder=3 if strong else 2,
                    )
                )

    calls = others["portcalls_container"].astype(float)
    sizes = 26 + 150 * (calls / calls.max())
    scatter = ax.scatter(
        others["x"], others["lat"], s=sizes, c=others["correlation"],
        cmap=HOUSE_CMAP, vmin=0.2, vmax=0.85, edgecolor="none",
        linewidth=0, alpha=0.95, zorder=4,
    )
    if not focus.empty:
        ax.scatter(
            focus["x"], focus["lat"],
            s=26 + 150 * float(focus["portcalls_container"].iloc[0]) / float(calls.max()),
            marker="o", color=COLORS["focus"], edgecolor="none",
            linewidth=0, zorder=6,
        )
        ax.annotate(
            "Shanghai", xy=(ox, oy), xytext=(9, 7), textcoords="offset points",
            fontsize=9, weight="bold", color=COLORS["focus_label"], zorder=8,
            path_effects=[path_effects.withStroke(linewidth=2.4,
                                                  foreground="white")],
        )

    ax.set_xlim(LON_CUT, LON_CUT + 360)
    ax.set_ylim(-48, 66)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_linewidth(0.5)
        ax.spines[side].set_color("#999999")

    place_labels(fig, ax, others)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["focus"],
               markeredgecolor="none", markersize=9, label="Shanghai"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9fb8cd",
               markeredgecolor="none", markersize=4,
               label="marker area $\propto$ container port calls"),
        Line2D([0], [0], color=COLORS["link_high"], linewidth=1.3, alpha=0.92,
               label="stronger"),
        Line2D([0], [0], color=COLORS["link_low"], linewidth=1.3, alpha=0.78,
               label="weaker"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.85,
              edgecolor="none", fontsize=8, borderpad=0.5)

    # Place the colorbar explicitly; an aspect-constrained axes strands it otherwise.
    ax.set_anchor("N")
    fig.canvas.draw()
    box = ax.get_position()
    cax = fig.add_axes([
        box.x0 + 0.30 * box.width, box.y0 - 0.075, 0.40 * box.width, 0.028,
    ])
    cbar = fig.colorbar(scatter, cax=cax, orientation="horizontal")
    cbar.set_label("Correlation with Shanghai (monthly log change in port calls)")
    cbar.ax.tick_params(labelsize=8)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base = FIGURE_DIR / "figure24_shanghai_vs_other_ports"
    fig.savefig(f"{base}.pdf", format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{base}.png", format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    return base


def main() -> None:
    monthly, lead = load_port_panel()
    corr = comovement(monthly)

    table = lead.copy()
    table["correlation"] = table["portname"].map(corr)
    coords = load_coordinates()
    table["lat"] = table["portid"].map(coords["lat"])
    table["lon"] = table["portid"].map(coords["lon"])

    focus_row = table[table["portname"].eq(FOCUS_PORT)]
    if focus_row.empty:
        raise ValueError("Shanghai missing from the lead-port table")
    flat, flon = float(focus_row["lat"].iloc[0]), float(focus_row["lon"].iloc[0])
    table["distance_km"] = [
        np.nan if pd.isna(la) else great_circle_km(flat, flon, la, lo)
        for la, lo in zip(table["lat"], table["lon"])
    ]

    conn = load_lsbci()
    table["connectivity"] = [
        conn.get(LSBCI_BY_ISO3.get(iso)) if iso in LSBCI_BY_ISO3
        else conn.get(LSBCI_PARTNER.get(c, c))
        for c, iso in zip(table["country"], table["ISO3"])
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = table[["country", "ISO3", "portname", "portcalls_container",
                 "correlation", "connectivity", "distance_km",
                 "lat", "lon"]].copy()
    out["months_used"] = len(monthly) - 1
    out.to_csv(REPORT_DIR / "shanghai_vs_other_ports.csv",
               index=False, encoding="utf-8-sig")

    missing = table.loc[table["lat"].isna(), "portname"].tolist()
    if missing:
        print(f"no coordinates, omitted from the map: {missing}")

    cut = table.loc[~table["portname"].eq(FOCUS_PORT), "connectivity"].median()
    print(f"link colour split at median LSBCI = {cut:.3f}")
    path = build_figure(table)
    print(f"months {len(monthly)}: {monthly.index.min().date()} -> "
          f"{monthly.index.max().date()}")
    print(f"countries plotted: {table['lat'].notna().sum()} of {len(table)}")
    miss = table.loc[table["connectivity"].isna() & table["lat"].notna(),
                     "country"].tolist()
    if miss:
        print(f"no LSBCI link drawn for: {miss}")
    print("\ncorrelation with Shanghai (top and bottom 5):")
    print(corr.head(5).round(3).to_string())
    print("  ...")
    print(corr.tail(5).round(3).to_string())
    print(f"\nfigure -> {path}.pdf")


if __name__ == "__main__":
    main()
