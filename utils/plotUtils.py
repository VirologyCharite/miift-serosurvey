import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import math
import string
import seaborn as sns
import matplotlib.lines as mlines


def annotateWithLetter(ax, letter, size=20, coords=(-0.05, 1)):
    """
    Annotate a C{matplotlib} axes with a C{letter} in the upper left corner of the plot.
    @param ax: A C{matplotlib} axes.
    @param letter: A C{str} specifying the letter to annotate C{ax} with.
    @param coords: The x, y coordinate to position the letter at.
    """
    x, y = coords
    ax.text(x, y, letter, transform=ax.transAxes, size=size, weight="bold")


def annotateWithLetters(axes, size=20, coords=(-0.05, 1)):
    """
    Annotate each C{matplotlib} axes in axes with a letter in alphabetical
    order.
    @param size: The font size of the letters.
    @param axes: An C{iterable} of C{matplotlib} axes.
    @param coords: The x, y coordinate for each figure to position the
        letters at.
    """
    # Give each subplot a letter (A, B, C...).
    for letter, ax in zip(string.ascii_uppercase, axes):
        annotateWithLetter(ax, letter, size, coords)


def computeTicks(x, step=5):
    """
    Computes x-ticks based on step size (C{step}).
    @param x: A list-like object of integers or floats.
    @param step: Tick frequency.
    """
    xMax, xMin = math.ceil(max(x)), math.floor(min(x))
    dMax, dMin = (
        xMax + abs((xMax % step) - step) + (step if (xMax % step != 0) else 0),
        xMin - abs((xMin % step)),
    )
    return range(dMin, dMax, step)


def hideGridX(ax):
    ax.grid(False, axis="x")


def hideGridY(ax):
    ax.grid(False, axis="y")


def reformatLegend(ax, legend, label, loc, ncol=2, border=True, fontSizePlot=12):
    framealpha = 0.7 if border else 0
    ax.legend(
        handles=legend[1:],
        ncol=ncol,
        title=label,
        columnspacing=1,
        borderaxespad=0.2,
        loc=loc,
        fontsize=fontSizePlot,
        title_fontsize=fontSizePlot,
        handleheight=0.5,
        handlelength=1.3,
        framealpha=framealpha,
    )


def removeLegend(ax):
    ax.legend_.remove()


def replaceLegend(ax, handles, loc="upper left"):
    """
    Replace the current legend in ax with a custom one.
    @param ax: A C{matplotlib} axes for plotting.
    @param handles: The handles you want to appear in the legend.
    @param loc: The location you want the legend to appear at.
    """
    try:  # Sometimes there is no legend.
        ax.legend_.remove()
    except AttributeError:
        pass
    ax.legend(handles=handles, loc=loc)


def addHDIlinesToPlot(
    ax,
    iData,
    varNames,
    color="black",
    yPosition=0,
    hdi=(0.03, 0.97),
    linewidth50=0.09,
    linewidth94=0.05,
    markersize=3.5,
):
    """
    Add highest posterior density intervals as horizontal lines to a ridge plot.
    @param ax: A C{matplotlib} axes containing a ridge plot representing the
    posterior distributions of paramater values estimated by MCMC.
    @param iData: An C{arviz.InferenceData} object.
    @param varNames: An C{iterable} specifying the variables (in the desired order)
    for which to add HDI lines to the plot.
    @param color: A C{str} specifying the color to use for plotting.
    @param yPosition: A C{float} specifying the y coordinate relative to the the
    original y-coordinate for plotting of the HDI lines. This is relevant if you want
    to plot HDI lines from multiple models in the same plot.
    @param hdi: A C{tuple} specifying the HDI to plot.
    """
    i = 0
    yTicks = ax.get_yticks()[::-1]
    post = iData.posterior
    iDataList = list(az.sel_utils.xarray_sel_iter(post, combined=True))
    iDataListOrdered = []
    for varName in varNames:
        for coord in iDataList:
            if varName in coord:
                iDataListOrdered.append(coord)

    for var, coords, _ in iDataListOrdered:
        yCoord = yTicks[i] + yPosition
        varPost = post[var].sel(coords).values.flatten()
        mean = varPost.mean()
        hdi50 = np.quantile(varPost, (0.25, 0.75))
        hdiPassed = np.quantile(varPost, hdi)

        ax.fill_between(
            hdiPassed,
            yCoord - linewidth94 / 2,
            yCoord + linewidth94 / 2,
            color=color,
            alpha=0.8,
            linewidth=0,
        )
        ax.fill_between(
            hdi50,
            yCoord - linewidth50 / 2,
            yCoord + linewidth50 / 2,
            color=color,
            alpha=0.8,
            linewidth=0,
        )
        ax.plot(
            mean,
            yCoord,
            marker="o",
            markersize=markersize,
            markeredgewidth=0.5,
            markeredgecolor="white",
            markerfacecolor=color,
        )
        i += 1


def ridgeForestPlot(
    iData,
    varNames,
    yTickLabels=None,
    figsize=(10, 10),
    fontsize=10,
    xlim=None,
    ax=None,
):
    """
    Make a mixture of a ridge and forest plot for displaying posterior distributions
    of parameters as determined by an MCMC run.
    @param iData: An C{arviz.InferenceData} object.
    @param varNames: An C{iterable} specifying the variables (in the desired order)
    whose posterior distributions should be plotted.
    @param yTickLabels: An C{iterable} of y-axis ticklabels to use.
    @param figsize: A C{tuple} specifying the size of the output figure.
    @param fontsize: The size of the font to appear in the plot.
    @param xlim: A C{tuple} or C{None} specifying which (if any) limits to use for
    the x-axis.
    @param ax: The C{matplotlib} axes to use for plotting.
    @return: A C{matplotlib.pyplot.axis} object containing a mix of a ridge and a
    forest plot for displaying posterior distributions.
    """
    assert varNames

    ax = ax or plt.subplots(1, 1, figsize=figsize)[1]
    az.plot_forest(
        iData,
        hdi_prob=1,
        combined=True,
        var_names=list(varNames),
        kind="ridgeplot",
        ridgeplot_truncate=False,
        ridgeplot_overlap=0.55,
        ridgeplot_alpha=0.3,
        markersize=3,
        linewidth=0,
        colors="blue",
        ax=ax,
    )
    leftLim, rightLim = xlim or ax.get_xlim()
    xTicks = []
    if abs(leftLim) >= 10 or abs(rightLim) >= 10:
        step = 4
    elif abs(leftLim) <= 2 or abs(rightLim) <= 2:
        step = 1
    else:
        step = 2
    xTickRange = range(-12, 13, step)
    for x in xTickRange:
        if leftLim <= x <= rightLim:
            linewidth = 0.15 if x else 0.5
            ax.vlines(x, ymin=-1, ymax=25, color="gray", linewidth=linewidth)
            xTicks.append(x)
    # Add 94% HDI lines as we know them from forest plots.
    addHDIlinesToPlot(ax, iData, varNames=varNames)
    if yTickLabels:
        ax.set_yticklabels(yTickLabels)
    ax.set_xticks(xTicks)
    ax.set_xlabel("Estimated parameter size")
    ax.set_xlim(leftLim, rightLim)
    ax.set_xticklabels(xTicks)
    ax.tick_params(axis="y", which="both", labelleft=False, labelright=True, size=10)
    setFontSize(ax, size=fontsize)

    plt.tight_layout()

    return ax


def forestPlotMultipleModels(
    iDataList,
    varNames,
    colors=None,
    legendLabels=None,
    yTickLabels=None,
    figsize=(10, 10),
    fontsize=10,
    xlim=None,
    legendLoc="upper left",
    linewidth50=0.09,
    linewidth94=0.05,
    markersize=3.5,
    yDistance=0.2,
    ax=None,
):
    """

    @param iDataList: A C{list} of C{az.InferenceData} objects.
    @param varNames: A C{iterable} of variable names that can be found in each
    C{az.InferenceData} object in C{iDataList}. The posterior distributions of these
    variables are plotted.
    @param colors: The color used for plotting each model.
    @param legendLabels: An C{iterable} of the labels specifying each model used in the
    plot's legend.
    @param yTickLabels: The labels supposed to be displayed on the y-axis,
    corresponding to the variable names in C{varNames}.
    @param figsize: The figure size.
    @param fontsize: The font size for C{yTickLabels}. The font size for C{
    legendLabels} is C{fontsize} - 2.
    @param xlim: The left and right limit of the x-axis.
    @param legendLoc: The location of the legend in the plot.
    @param ax: The C{matplotlib} axes object used for plotting.
    @return: A C{matplotlib} axes object showing posterior parameter distributions
    from several fitted models.
    """
    assert varNames

    colors = colors or sns.color_palette("husl", len(iDataList)).as_hex()

    assert len(colors) == len(iDataList)
    if legendLabels:
        assert len(legendLabels) == len(iDataList)

    ax = ax or plt.subplots(1, 1, figsize=figsize)[1]
    # This is a bit of a dummy step, just so we obtain appropriate y-axis offsets.
    # The actual plotting is done below
    az.plot_forest(
        iDataList[0],
        hdi_prob=0.94,
        combined=True,
        var_names=list(varNames),
        markersize=0,
        linewidth=0,
        colors=colors[0],
        ax=ax,
    )
    leftLim, rightLim = xlim or ax.get_xlim()
    xTicks = []
    if abs(leftLim) >= 10 or abs(rightLim) >= 10:
        step = 4
    elif abs(leftLim) <= 2 or abs(rightLim) <= 2:
        step = 1
    else:
        step = 2
    xTickRange = range(-12, 13, step)
    for x in xTickRange:
        if leftLim <= x <= rightLim:
            linewidth = 0.15 if x else 0.5
            ax.vlines(x, ymin=-1, ymax=25, color="gray", linewidth=linewidth)
            xTicks.append(x)
    # Add 94% HDI lines as we know them from forest plots.
    yPosition = -yDistance
    for color, iData in zip(colors[::-1], iDataList[::-1]):
        addHDIlinesToPlot(
            ax,
            iData,
            varNames=varNames,
            color=color,
            yPosition=yPosition,
            linewidth50=linewidth50,
            linewidth94=linewidth94,
            markersize=markersize,
        )
        yPosition += yDistance
    if yTickLabels:
        ax.set_yticklabels(yTickLabels)
    ax.set_xticks(xTicks)
    ax.set_xlabel("Estimated parameter size")
    ax.set_xlim(leftLim, rightLim)
    ax.set_xticklabels(xTicks)
    ax.tick_params(axis="y", which="both", labelleft=False, labelright=True, size=10)
    setFontSize(ax, size=fontsize)
    legendLines = []
    if legendLabels:
        for color, label in zip(colors, legendLabels):
            legendLine = mlines.Line2D(
                [],
                [],
                color=color,
                marker="o",
                markersize=5,
                linewidth=2,
                label=label,
                markeredgecolor="white",
            )
            legendLines.append(legendLine)

        plt.legend(legendLines, legendLabels, loc=legendLoc, fontsize=fontsize - 2)
    plt.tight_layout()

    return ax


def setFontSize(ax, size=15):
    """
    Change the font size of all texts in a C{matplolib} axis.
    @param ax: A C{matplotlib} axes for plotting.
    @param size: The font size you want the text in ax to have.
    """
    texts = (
        [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    )

    legend = ax.get_legend()
    if legend:
        texts.append(legend.get_texts())
        texts.append(legend.get_title())

    for text in texts:
        if isinstance(text, list):
            for ele in text:
                ele.set_fontsize(size)
        else:
            text.set_fontsize(size)

    ax.title.set_fontsize(size + 2)


def boldStr(text):
    """
    @param text: A C{str}.
    @return: text in a bold font.
    """
    return r"$\bf{" + str(text) + "}$"


def returnOrderJacksonTali():
    return [
        "adult",
        "male",
        "migration",
        "cave",
        "insectivorous",
        "yangochiroptera",
        "colony",
    ]


def returnColorDict(df):

    colorsSpecies = dict(
        zip(
            df.species.unique(),
            sns.color_palette("Spectral", n_colors=len(df.species.unique())),
        )
    )
    colorsGenus = dict(
        zip(
            df.genus.unique(),
            sns.color_palette("cubehelix", n_colors=len(df.genus.unique()))[::-1],
        )
    )
    colorsCountry = dict(
        zip(
            df.country.unique(),
            sns.color_palette("tab10", n_colors=len(df.country.unique())),
        )
    )

    genus2Cave = {
        genus: cave for genus, cave in df[["genus", "cave"]].value_counts().index
    }
    colorsCave = {
        genus: "red" if genus2Cave[genus] else "blue" for genus in df.genus.unique()
    }

    genus2Migration = {
        genus: migration
        for genus, migration in df[["genus", "migration"]].value_counts().index
    }
    colorsMigration = {
        genus: "red" if genus2Migration[genus] else "blue"
        for genus in df.genus.unique()
    }

    genus2Insectivorous = {
        genus: insectivorous
        for genus, insectivorous in df[["genus", "insectivorous"]].value_counts().index
    }
    colorsInsectivorous = {
        genus: "red" if genus2Insectivorous[genus] else "blue"
        for genus in df.genus.unique()
    }

    colorsRandSubgroups = dict(
        zip(
            range(13),
            sns.color_palette("cubehelix", n_colors=len(df.genus.unique()))[::-1],
        )
    )

    return {
        "species": colorsSpecies,
        "genus": colorsGenus,
        "country": colorsCountry,
        "cave": colorsCave,
        "migration": colorsMigration,
        "insectivorous": colorsInsectivorous,
        "random": colorsRandSubgroups,
    }


def returnTropicalViruses():
    return ("denv2", "yfv", "wnv", "mumps", "piv2", "chikv", "veev")


def returnViruses():
    return (
        "denv2",
        "yfv",
        "wnv",
        "measles",
        "mumps",
        "piv1",
        "piv2",
        "chikv",
        "veev",
        "hcov-229e",
        "sars-cov",
        "rvfv",
        "rsv",
    )

def returnVirusesAll():
    return (
        "denv2",
        "yfv",
        "wnv",
        "measles",
        "mumps",
        "piv1",
        "piv2",
        "chikv",
        "veev",
        "hcov-229e",
        "sars-cov",
        "rvfv",
        "rsv",
        "dobv",
    )


def returnVirusesLabelDict():
    return {
        "denv2": "DENV-2",
        "yfv": "YFV",
        "wnv": "WNV",
        "measles": "MeV",
        "mumps": "MuV",
        "piv1": "PIV-1",
        "piv2": "PIV-2",
        "chikv": "CHIKV",
        "veev": "VEEV",
        "hcov-229e": "hCoV-229E",
        "sars-cov": "SARS-CoV",
        "rvfv": "RVFV",
        "rsv": "RSV",
        "dobv": "DOBV",
    }
