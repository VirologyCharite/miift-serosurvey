import pandas as pd
import numpy as np
from pathlib import Path
from functools import partial
from pandas.api.types import is_string_dtype, is_numeric_dtype

TOP_LEVEL_DIR = Path("../data")
DATA_FILE = TOP_LEVEL_DIR / "data_complete.xlsx"
DATA_FILE_COLONY_LIFESPAN = TOP_LEVEL_DIR / "bat_colony_size_n_lifespan_data.xlsx"


def toBinaryCol(df, colOrig, colNew, referenceCat, nanTerm="nd"):
    maskNan = df[colOrig] == nanTerm
    df[colNew] = (df[colOrig] == referenceCat).astype(int)
    df.loc[maskNan, colNew] = pd.NA


# Lifespan imputation.
# In the multilevel models, we will use a lognormal distribution for modelling the
# expected lifespan (since it is not known for all species). For 5 species, lifespan is
# known (however uncertain), so we will use a relatively small standard deviation (0.2)
# in the log normal distribution (and assume the mean to correspond to the known
# lifespan). For species with unknown lifespan, we treat the expected lifespan of a
# species from the same genus as the mean and use a wider standard deviation (0.3).
# For a species with no corresponding genus but a corresponding family, we use the
# family's mean lifespan as the new mean and a standard deviation of 0.4. Note that in
# our dataset each genus only has one species with available lifespan info. For species
# with a corresponding genus that has no lifespan info at all, we assume a mean of 12
# years with a standard deviation of 0.6. Standard deviations were set according to
# manual exploration of different values (see stats_analyses notebook).

# For computing mu values of a log-normal distribution from given mean and sigma, see:
# https://brilliant.org/wiki/log-normal-distribution/.
#
# **Mode of a lognormal distribution**
# x=e(mu - sigma^2)
# **With x being lifespan (the mode), solve for mu**
# ln(x) = mu - sigma^2
# mu = ln(x) + sigma^2
# **Mean of a lognormal distribution**
# x = e(mu + sigma^2 / 2)
# **With x being lifespan (the mean), solve for mu**
# ln(x) = mu + sigma^2 / 2
# mu = ln(x) - sigma^2 / 2


def lognormalMuFromMean(mean, sigma):
    return np.log(mean) - sigma**2 / 2


def lognormalMuFromMode(mode, sigma):
    return np.log(mode) + sigma**2


def returnLifespanDict(meanLifespanNoInfo=12):
    meanLifespanNoInfo = meanLifespanNoInfo
    sdYearsNoInfo = 0.6
    sdYearsFamily = 0.4
    sdYearsGenus = 0.3
    sdYearsSpecies = 0.2

    # Values are taken from bat_colony_size_n_lifespan_data.xlsx
    lifespanDict = {  # Species with "known" average lifespans.
        "aegyptiacus": {"mean": 9, "max": 22, "sd": sdYearsSpecies},
        "helvum": {"mean": 15, "max": 22, "sd": sdYearsSpecies},
        "daubentonii": {"mean": 4.5, "max": 28, "sd": sdYearsSpecies},
        "noctula": {"mean": 8, "max": None, "sd": sdYearsSpecies},
        "jamaicensis": {"mean": 8, "max": 10, "sd": sdYearsSpecies},
        "sp.": {"mean": 6, "max": 7, "sd": sdYearsSpecies},
        # Species with unknown average and maximum lifespans, values are inferred from
        # species of the same genus.
        "lituratus": {"mean": 8, "sd": sdYearsGenus},
        # Species with unknown average and maximum lifespans, values are inferred from
        # species of the same family.
        "franqueti": {"mean": 12, "sd": sdYearsFamily},
        "monstrosus": {"mean": 12, "max": 30, "sd": sdYearsFamily},
        "pusillus": {"mean": 12, "sd": sdYearsFamily},
        "torquata": {"mean": 12, "max": 30, "sd": sdYearsFamily},
        "dasycneme": {"mean": 6.25, "max": 20, "sd": sdYearsFamily},
        # Since caffer and gigas seem to be closely related to rhinolophidae (sp.),
        # see https://de.wikipedia.org/wiki/Rundblattnasen, we assign
        # the same mean value here but a wider standard deviation.
        "caffer": {"mean": 6, "sd": sdYearsNoInfo},
        "gigas": {
            "mean": 6,
            "sd": sdYearsNoInfo,
        },
        # Species with no information at all.
        "afra": {"mean": meanLifespanNoInfo, "sd": sdYearsNoInfo},
        "inflatus": {"mean": meanLifespanNoInfo, "sd": sdYearsNoInfo},
    }
    return lifespanDict


def _returnLifespan(species, param, meanLifespanNoInfo=12):
    lifespanDict = returnLifespanDict(meanLifespanNoInfo=meanLifespanNoInfo)
    return lifespanDict.get(species)[param]


def returnLifespanMean(species, meanLifespanNoInfo=12):
    return _returnLifespan(species, param="mean", meanLifespanNoInfo=meanLifespanNoInfo)


def returnLifespanMax(species):
    try:
        return _returnLifespan(species, param="max")
    except KeyError:
        return pd.NA


def returnLifespanSd(species):
    return _returnLifespan(species, param="sd")


def completeLifespans(df, meanLifespanNoInfo=12):
    returnLifespanMeanWithArg = partial(
        returnLifespanMean, meanLifespanNoInfo=meanLifespanNoInfo
    )

    # Apply the new function
    df["lifespanMean"] = df.species.apply(returnLifespanMeanWithArg)
    df["lifespanSd"] = df.species.apply(returnLifespanSd)


def createColonysizeDf(df):
    # Group by sampling site, year and species to estimate colonysizes separately for
    # the resulting groups.
    dfColonysize = (
        df.groupby(
            [
                "year",
                "site",
                "species",
                "logColonysizeSd",
                "logColonysizeMin",
                "logColonysizeMax",
            ]
        )
        .logColonysizeMean.unique()
        .reset_index()
    )
    dfColonysize["logColonysizeMean"] = dfColonysize.logColonysizeMean.str[0].astype(
        float
    )
    # Sort by colonysize (important for later).
    dfColonysize = dfColonysize.sort_values(by="logColonysizeMean").reset_index()
    dfColonysize["sampleGroupColony"] = dfColonysize.index

    df = df.merge(dfColonysize, on=["year", "site", "species"], suffixes=["", "_y"])
    assert len(dfColonysize) == df.sampleGroupColony.nunique()

    return df, dfColonysize


def createSpeciesDf(df):
    # Group by species to estimate lifespan separately for each species.
    dfSpecies = (
        df.groupby("species")[["lifespanMean", "lifespanSd"]]
        .value_counts()
        .reset_index()[["species", "lifespanMean", "lifespanSd"]]
    )
    # We are subtracting 1 from the mean expected life span here because in the log
    # normal distribution in regression.py (function fitMultilevelB) we are
    # adding 1 to make sure that the minimum lifespan is 1 year.
    dfSpecies["lifespanMu"] = dfSpecies.apply(
        lambda row: lognormalMuFromMean(
            mean=row["lifespanMean"] - 1, sigma=row["lifespanSd"]
        ),
        axis=1,
    )
    dfSpecies["lifespanSigma"] = dfSpecies.lifespanSd
    dfSpecies["speciesCode"] = dfSpecies.index

    df = df.merge(dfSpecies, on="species", suffixes=["", "_y"])
    assert len(dfSpecies) == df.speciesCode.nunique()

    return df, dfSpecies


def returnLogColonysizesNormalDistDict():
    # The following values are derived from
    # ../Serology2023/bat_colony_size_data_with_sources:
    # Distributions for the different genera (log scale):
    #     - coleura(afra): Normal(6, 2)
    #     - hipposideros(caffer): Normal(8, 3)
    #     - macronycteris(gigas): Normal(4.7, 0.7)
    #     - epomops(franqueti): Normal(1.15, 0.6)
    #     - hypsignathus(monstrosus): Normal(1.5, 1)
    #     - micropteropus(pusillus): Normal(0.7, 0.8)
    #     - rousettus(aegyptiacus): Normal(6, 1.5)
    #     - rhinolophus(sp.): Normal(1.2, 0.6)
    #     - miniopterus(inflatus): Normal(6.4, 2)
    #     - eidolon(helvum): Normal(12.6, 0.5)
    #     - myotis(dasycneme): Normal(3.6, 1.5)
    #     - myotis(daubentonii): Normal(2.5, 1.25)
    #     - nyctalus(noctula): Normal(3, 1.5)
    #     - artibeus(jamaicensis): Normal(2.3, 0.35)
    #     - artibeus(lituratus): Normal(2, 0.5)
    #     - myonycteris(torquata): Normal(0, 0.01)
    #
    # Following https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0006367#pone.0006367-Taylor1
    # (sampling was done in 2008, sampling for our data was done 2009 and 2010) we set the mean to log(400,000)
    # for **eidolon helvum** and an sd of 0.3.

    return {
        "afra": {"mean": 6, "sd": 2},
        "caffer": {"mean": 8, "sd": 3},
        "gigas": {"mean": 4.7, "sd": 0.7},
        "franqueti": {"mean": 1.15, "sd": 0.6},
        "monstrosus": {"mean": 1.5, "sd": 1},
        "pusillus": {"mean": 0.7, "sd": 0.8},
        "aegyptiacus": {"mean": 6, "sd": 1.5},
        "sp.": {"mean": 2, "sd": 1},
        # "sp.": {"mean": 2, "sd": 1},
        "inflatus": {"mean": 6.4, "sd": 2},
        "helvum": {"mean": 12.9, "sd": 0.3},
        "dasycneme": {"mean": 3.6, "sd": 1.5},
        "daubentonii": {"mean": 2.5, "sd": 1.25},
        "noctula": {"mean": 3, "sd": 1.5},
        "jamaicensis": {"mean": 2.3, "sd": 0.35},
        "lituratus": {"mean": 2, "sd": 0.5},
        "torquata": {"mean": 0, "sd": 0.5},
    }


def _returnColonysize(species, param):
    logColonySizesNormalDistDict = returnLogColonysizesNormalDistDict()
    return logColonySizesNormalDistDict[species][param]


def returnColonysizeMean(species):
    return _returnColonysize(species, param="mean")


def returnColonysizeSd(species):
    return _returnColonysize(species, param="sd")


def addColonysizesNLifespans(df, meanLifespanNoInfo=12):
    dataColonysizes = DATA_FILE_COLONY_LIFESPAN
    dfColonysizes = pd.read_excel(dataColonysizes, sheet_name="Tabelle2")

    colsColonysize = [
        "colonysizeMin",
        "colonysizeAvrgMin",
        "colonysizeAvrg",
        "colonysizeAvrgMax",
        "colonysizeMax",
    ]
    # We use maximum colony size of 5 here so that the statistical model isn't too
    # strongly influenced by this "outlier". Also, an at least temporary colony size of
    # 5 does not seem too unrealistic.
    dfColonysizes.loc[dfColonysizes.species == "torquata", "colonysizeMax"] = 5

    logColnames = [
        f"log{colname[0].capitalize() + colname[1:]}" for colname in colsColonysize
    ]
    dfColonysizes.loc[dfColonysizes.colonysizeMin.isna(), "colonysizeMin"] = 1
    dfColonysizes[logColnames] = np.log(dfColonysizes[colsColonysize])
    dfColonysizes.loc[dfColonysizes.logColonysizeAvrg.isna(), "logColonysizeAvrg"] = (
        dfColonysizes["logColonysizeAvrgMin"] + dfColonysizes["logColonysizeAvrgMax"]
    ) / 2
    dfColonysizes.loc[dfColonysizes.logColonysizeAvrg.isna(), "logColonysizeAvrg"] = (
        dfColonysizes["logColonysizeMin"] + dfColonysizes["logColonysizeMax"]
    ) / 2

    dfColonysizes["logColonysizeMin"] = dfColonysizes["logColonysizeMin"].round(2)
    dfColonysizes["logColonysizeMax"] = dfColonysizes["logColonysizeMax"].round(2)

    logColonySizesNormalDistDict = returnLogColonysizesNormalDistDict()

    assert all(
        species in logColonySizesNormalDistDict for species in df.species.unique()
    )
    df["logColonysizeMean"] = df.species.apply(returnColonysizeMean)
    df["logColonysizeSd"] = df.species.apply(returnColonysizeSd)
    assert all(
        species in df.species.unique() for species in dfColonysizes.species.unique()
    )
    solitarySpecies = df[df.colony == 0].species.unique()
    colonySpecies = df[df.colony == 1].species.unique()

    # The threshold corresponds to a mean colony size of about 5 (~1.5 on the log
    # scale). We don't want to count species with a maximum colony size <= 12 as
    # colony-dwelling.
    colonyThreshold = 1.5
    errorMsg1 = (
        f"All log colony sizes for solitary bats should be below {colonyThreshold}: "
        + ", ".join(
            f"{species}: {logColonySizesNormalDistDict[species]['mean']}"
            for species in sorted(solitarySpecies)
        )
    )
    errorMsg2 = (
        f"All log colony sizes for colony-dwelling bats should be above "
        f"{colonyThreshold}: "
        + ", ".join(
            f"{species}: {logColonySizesNormalDistDict[species]['mean']}"
            for species in sorted(colonySpecies)
        )
    )
    assert all(
        logColonySizesNormalDistDict[species]["mean"] < colonyThreshold
        for species in solitarySpecies
    ), errorMsg1
    assert all(
        logColonySizesNormalDistDict[species]["mean"] >= colonyThreshold
        for species in set(colonySpecies)
    ), errorMsg2

    df = pd.merge(df, dfColonysizes, on="species", how="left", suffixes=("", "_y"))

    lifespanDict = returnLifespanDict(meanLifespanNoInfo=meanLifespanNoInfo)
    assert all(species in df.species.unique() for species in lifespanDict)

    returnLifespanMeanWithArg = partial(
        returnLifespanMean, meanLifespanNoInfo=meanLifespanNoInfo
    )
    df["lifespanMean"] = df.species.apply(returnLifespanMeanWithArg)
    df["lifespanMax"] = df.species.apply(returnLifespanMax)

    return df


def processColumnNamesAndValues(df, replaceColsVirusBinary, allAnimals=False):
    # Binary virus columns
    replaceColsOther = {
        "Typical colony size": "colonysize",
        "Mating Season?": "matingseason",
        "Date sampling": "date",
        "Typical lifespan": "lifespan",
    }
    df = df.rename(columns=replaceColsVirusBinary).rename(columns=replaceColsOther)

    # All columns to lowercase
    replaceColsRemaining = {col: col.lower() for col in df.columns}
    df = df.rename(columns=replaceColsRemaining)

    # All values to lowercase
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            continue
        if allAnimals:
            if col not in  ("matingseason", "site", "year", "date", "season",
                            "migration"):
                assert not any(pd.isna(df[col])), f"Column {col} has NaN values."
        else:
            assert not any(pd.isna(df[col])), f"Column {col} has NaN values."

        if is_string_dtype(df[col]):
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(lambda value: value.lower())

    # Turn not determined into NaNs.
    df = df.replace(
        {
            "age": {"nd": pd.NA},
            "migration": {"n": 0, "y": 1},
            "date": {"nd": pd.NA, "ND": pd.NA},
        }
    )

    df["date"] = pd.DatetimeIndex(df.date)
    df["month"] = df.date.dt.month

    return df


def addBinaryColonyInfo(df):
    speciesToColony = {
        "aegyptiacus": 1.0,
        "afra": 1.0,
        "caffer": 1.0,
        "dasycneme": 1.0,
        "daubentonii": 1.0,
        "franqueti": 0.0,
        "gigas": 1.0,
        "helvum": 1.0,
        "inflatus": 1.0,
        "jamaicensis": 1.0,
        "lituratus": 1.0,
        "monstrosus": 1.0,
        "noctula": 1.0,
        "not determined": pd.NA,
        "pusillus": 0.0,
        "sp.": 0.0,
        "torquata": 0.0,
    }
    df["colony"] = df.species.map(speciesToColony)

    # Correct samples that have a wrong "colony" label.
    colonySpecies = {
        "gb405",
        "gb393",
        "gb394",
        "gb395",
        "gb396",
        "gb397",
        "gb398",
        "gb399",
        "gb400",
        "gb401",
        "gb402",
        "gb404",
        "gb406",
        "gb407",
        "gb1112",
        "gb403",
    }
    df.loc[df["sample no."].isin(colonySpecies), "colony"] = 1


def addBatFamily(df):
    genusToFamily = {
        "artibeus": "phyllostomidae",
        "coleura": "emballonuridae",
        "eidolon": "pteropodidae",
        "epomops": "pteropodidae",
        "hipposideros": "hipposideridae",
        "macronycteris": "hipposideridae",
        "hypsignathus": "pteropodidae",
        "micropteropus": "pteropodidae",
        "miniopterus": "miniopteridae",
        "myonycteris": "pteropodidae",
        "myotis": "vespertilionidae",
        "nyctalus": "vespertilionidae",
        "rhinolophus": "rhinolophidae",
        "rousettus": "pteropodidae",
    }
    df["family"] = df.genus.map(genusToFamily)


def addBinaryColumns(df):
    toBinaryCol(df, "climate zone", "tropical", "tropical")
    toBinaryCol(df, "roosting site", "cave", "c")
    toBinaryCol(df, "sex", "male", "m")
    toBinaryCol(df, "age", "adult", "a")
    toBinaryCol(df, "dietry", "insectivorous", "i", nanTerm="o")
    toBinaryCol(df, "matingseason", "matingseason", "yes")
    df["adult2"] = df.adult.isin(("a", "sa")).astype(int)


# Data preprocessing.
def loadData(animal="bat", meanLifespanNoInfo=12):
    assert animal in ("bat", "rodent", "all")
    dfOrig = pd.read_excel(DATA_FILE, skiprows=1)
    replaceColsVirusBinary = {
        col: col.replace(" IIFT", "")
        for col in dfOrig.columns
        if "IIFT" in col and not "Date" in col
    }
    viruses = list(replaceColsVirusBinary.values())
    virusesLowerCase = [virus.lower() for virus in viruses]

    columnsToDrop = [
        "Mating system",
        "Sampled by",
        "Body length (mm)",
        "Tail length (mm)",
        "Sex_baby",
        "AB_baby (mm)",
        "Body length_baby (mm)",
        "Tail length_baby (mm)",
        "Parasite",
        "Comment",
    ]
    if animal == "all":
        dfOrig = dfOrig.drop(columns=columnsToDrop).copy()
        df = processColumnNamesAndValues(dfOrig, replaceColsVirusBinary, allAnimals=True)
        dfTropical = None
    else:
        dfOrig = dfOrig[dfOrig.Animal == animal].drop(columns=columnsToDrop).copy()
        df = processColumnNamesAndValues(dfOrig, replaceColsVirusBinary)
        addBinaryColonyInfo(df)
        addBatFamily(df)
        df = addColonysizesNLifespans(df, meanLifespanNoInfo=meanLifespanNoInfo)
        addBinaryColumns(df)
        df["yearSite"] = df["year"].astype(str) + df["site"]
        df["yearSiteSpecies"] = df["year"].astype(str) + df["site"] + df["species"]
        df["region"] = df.country.replace(
            {"gabon": "gabon_congo", "republic of congo": "gabon_congo", })
        completeLifespans(df, meanLifespanNoInfo=meanLifespanNoInfo)
        df["infections_total"] = (df[[virus for virus in virusesLowerCase]]).sum(axis=1)
        dfTropical = df[df.tropical == 1].copy().reset_index()
        completeLifespans(dfTropical, meanLifespanNoInfo=meanLifespanNoInfo)

    return dfOrig, df, dfTropical


def preprocessDfModelSimple(df):
    featuresSimple = [
        "yangochiroptera",
        "migration",
        "insectivorous",
        "cave",
        "logColonysizeMean",
        "adult",
        "male",
    ]
    dfSimple = df.dropna(subset=featuresSimple)
    return dfSimple


def preprocessDfModelB(df):
    dfMultilevelB = df.copy()
    families = [
        "emballonuridae",
        "hipposideridae",
        "miniopteridae",
        "phyllostomidae",
        "pteropodidae",
        "rhinolophidae",
    ]
    dfMultilevelB["familyCode"] = pd.Categorical(
        dfMultilevelB.family, categories=families
    ).codes
    dfMultilevelB["yearSiteCode"], yearSite = pd.factorize(dfMultilevelB.yearSite)
    dfMultilevelB["regionYear"] = dfMultilevelB["region"] + dfMultilevelB[
        "year"
    ].astype("str")
    regions = ["gabon_congo", "ghana", "panama"]
    dfMultilevelB["regionCode"] = pd.Categorical(
        dfMultilevelB["region"], categories=regions
    ).codes
    regionYearCombinations = [
        "gabon_congo2003",
        "gabon_congo2005",
        "gabon_congo2006",
        "gabon_congo2008",
        "gabon_congo2009",
        "ghana2009",
        "ghana2010",
        "panama2011",
    ]
    dfMultilevelB["regionYearCode"] = pd.Categorical(
        dfMultilevelB["regionYear"], categories=regionYearCombinations
    ).codes

    featuresMultilevelB = [
        "migration",
        "insectivorous",
        "cave",
        "logColonysizeMean",
        "male",
        "adult",
        "lifespanMean",
        "family",
        "region",
    ]
    dfMultilevelB = dfMultilevelB.dropna(subset=featuresMultilevelB)

    return dfMultilevelB
