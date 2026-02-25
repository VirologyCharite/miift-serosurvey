import argparse
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Plot a world map highlighting sample countries.")
parser.add_argument("shapefile", help="Path to the Natural Earth shapefile (.shp)")
args = parser.parse_args()

SHAPEFILE = args.shapefile

HIGHLIGHT_COUNTRIES = {
    "Gabon",
    "Ghana",
    "Republic of Congo",
    "Germany",
    "Netherlands",
    "Panama",
    "Thailand",
}

COLOR_DEFAULT = "#AAAAAA"      # gray for all other countries
COLOR_HIGHLIGHT = "#8B0000"    # dark red for sample countries
COLOR_BORDER = "#FFFFFF"       # white borders
COLOR_OCEAN = "#FFFFFF"        # light blue background

# --- Load data ---
world = gpd.read_file(SHAPEFILE)

# Natural Earth uses the NAME column; check which names match
# Uncomment the next line to inspect available country names if needed:
# print(world["NAME"].sort_values().to_list())

# Natural Earth uses "Congo" for Republic of Congo (Brazzaville)
# and "Dem. Rep. Congo" for DRC. Adjust the mapping below if needed.
name_overrides = {
    "Republic of Congo": "Congo",      # NE name for Republic of Congo
    "Netherlands": "Netherlands",      # NE name (not "The Netherlands")
}

# Build a resolved set of NE names to highlight
highlight_ne_names = set()
for name in HIGHLIGHT_COUNTRIES:
    highlight_ne_names.add(name_overrides.get(name, name))

# Assign colors
world["color"] = world["NAME"].apply(
    lambda n: COLOR_HIGHLIGHT if n in highlight_ne_names else COLOR_DEFAULT
)

# --- Plot ---
fig, ax = plt.subplots(figsize=(14, 7))

world.plot(
    ax=ax,
    color=world["color"],
    edgecolor=COLOR_BORDER,
    linewidth=0.4,
)

ax.set_facecolor(COLOR_OCEAN)
fig.patch.set_facecolor(COLOR_OCEAN)

ax.set_axis_off()
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)

# Legend
legend_handles = [
    mpatches.Patch(facecolor=COLOR_HIGHLIGHT, edgecolor="#555555", linewidth=0.5,
                   label="Sample countries"),
    mpatches.Patch(facecolor=COLOR_DEFAULT, edgecolor="#555555", linewidth=0.5,
                   label="Other countries"),
]
ax.legend(
    handles=legend_handles,
    loc="lower left",
    frameon=True,
    framealpha=0.9,
    fontsize=9,
)

plt.tight_layout()
plt.savefig("world_map.pdf", dpi=1200, bbox_inches="tight")
plt.savefig("world_map.png", dpi=1200, bbox_inches="tight")
print("Saved world_map.pdf and world_map.png")