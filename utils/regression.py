import pymc as pm
import numpy as np
import arviz as az, xarray

# The following is a fix for a problem related to upgrading my Mac OS
# https://discourse.pymc.io/t/pytensor-fails-to-compile-model-after-upgrading-to-mac-os-15-4/16796/7
import pytensor

pytensor.config.cxx = "/usr/bin/clang++"
from pathlib import Path

from utils.dataUtils import createSpeciesDf, createColonysizeDf

RANDOM_SEED = 10
RESULTS_DIR = Path("..", "results")
RESULTS_DIR_MODEL_A = RESULTS_DIR / "modelA"
RESULTS_DIR_MODEL_B = RESULTS_DIR / "modelB"
IDATA_DIR_MODEL_A = RESULTS_DIR_MODEL_A / "iData"
IDATA_DIR_MODEL_B = RESULTS_DIR_MODEL_B / "iData"


# Model A
def fitSimple(virus, df, prior_predictive=False, target_accept=0.9, studentt=False):
    outputFile = IDATA_DIR_MODEL_A / f"{virus}.nc"

    def normalOrStudentDist(name, mu, sigma, studentt):
        return (
            pm.StudentT(name, mu=mu, sigma=sigma, nu=4)
            if studentt
            else pm.Normal(name, mu=mu, sigma=sigma)
        )

    sigma = 0.5
    sigmaColonysize = 0.2
    df = df.dropna(subset=[virus])

    df, dfColony = createColonysizeDf(df)
    df, dfSpecies = createSpeciesDf(df)
    # For the solitary bats, we have an sd of 0.5 on the log scale but a maximum value
    # of 0. To solve that issue, we set the maximum value to np.log(4), allowing for
    # a maximum number of 4 animals.
    dfColony.loc[dfColony.logColonysizeMax == 0, "logColonysizeMax"] = np.log(4)

    with pm.Model() as m:
        intercept = normalOrStudentDist("intercept", mu=0, sigma=1.5, studentt=studentt)
        b_yangochiroptera = normalOrStudentDist(
            "yangochiroptera", mu=0.0, sigma=sigma, studentt=studentt
        )
        b_migration = normalOrStudentDist(
            "migration", mu=0.0, sigma=sigma, studentt=studentt
        )
        b_insectivorous = normalOrStudentDist(
            "insectivorous", mu=0.0, sigma=sigma, studentt=studentt
        )
        b_cave = normalOrStudentDist("cave", mu=0.0, sigma=sigma, studentt=studentt)
        b_male = normalOrStudentDist("male", mu=0.0, sigma=sigma, studentt=studentt)
        b_adult = normalOrStudentDist("adult", mu=0.0, sigma=sigma, studentt=studentt)
        b_logColonysize = normalOrStudentDist(
            "colonysize", 0.0, sigmaColonysize, studentt=studentt
        )

        yangochiroptera = pm.intX(df.yangochiroptera)
        migration = pm.intX(df.migration)
        insectivorous = pm.intX(df.insectivorous)
        cave = pm.intX(df.cave)
        male = pm.intX(df.male)
        adult = pm.intX(df.adult)
        sampleGroupColony = pm.intX(df.sampleGroupColony)

        # The following variables are modeled assuming measurement uncertainty.
        log_colonysize_true = pm.TruncatedNormal(
            "log_colonysize_true",
            mu=dfColony.logColonysizeMean,
            sigma=dfColony.logColonysizeSd,
            lower=dfColony.logColonysizeMin,
            upper=dfColony.logColonysizeMax,
            shape=len(dfColony),
        )

        mu = (
            intercept
            + b_yangochiroptera * yangochiroptera
            + b_migration * migration
            + b_insectivorous * insectivorous
            + b_cave * cave
            + b_logColonysize * log_colonysize_true[sampleGroupColony]
            + b_adult * adult
            + b_male * male
        )
        p_bar = pm.Deterministic("p_bar", pm.math.invlogit(mu))

        # Bernoulli random vector with probability of success
        # given by sigmoid function and actual data as observed
        pm.Bernoulli(name="seroPos", p=p_bar, observed=df[virus])

        if prior_predictive:
            iDataCurr = pm.sample_prior_predictive(
                random_seed=RANDOM_SEED, return_inferencedata=True, var_names=["p_bar"]
            )
        else:
            if outputFile.exists():
                iDataCurr = az.from_netcdf(outputFile)
            else:
                iDataCurr = pm.sample(
                    random_seed=RANDOM_SEED,
                    return_inferencedata=True,
                    target_accept=target_accept,
                    draws=4000,
                    tune=2000,
                    log_likelihood=True,
                    # nuts_sampler="numpyro",
                )
                # Add variable for colonysize in units of 5
                iDataCurr.posterior["colonysize_5units"] = (
                    iDataCurr.posterior["colonysize"] * 5
                )
                az.to_netcdf(iDataCurr, outputFile)

        return m, iDataCurr

# Model B
def variableTransformationModelB(iData):
    iData.posterior["lifespan_5years"] = iData.posterior["lifespan"] * 5
    iData.posterior["colonysize_5units"] = iData.posterior["colonysize"] * 5

    region_gabon_congo = iData.posterior.sel(region_dim_0=0)["region"]
    region_ghana = iData.posterior.sel(region_dim_0=1)["region"]
    region_panama = iData.posterior.sel(region_dim_0=2)["region"]

    region_year_gabon_congo2003 = (
        region_gabon_congo + iData.posterior.sel(region_year_dim_0=0)["region_year"]
    )
    region_year_gabon_congo2005 = (
        region_gabon_congo + iData.posterior.sel(region_year_dim_0=1)["region_year"]
    )
    region_year_gabon_congo2006 = (
        region_gabon_congo + iData.posterior.sel(region_year_dim_0=2)["region_year"]
    )
    region_year_gabon_congo2008 = (
        region_gabon_congo + iData.posterior.sel(region_year_dim_0=3)["region_year"]
    )
    region_year_gabon_congo2009 = (
        region_gabon_congo + iData.posterior.sel(region_year_dim_0=4)["region_year"]
    )
    region_year_ghana2009 = (
        region_ghana + iData.posterior.sel(region_year_dim_0=5)["region_year"]
    )
    region_year_ghana2010 = (
        region_ghana + iData.posterior.sel(region_year_dim_0=6)["region_year"]
    )
    region_year_panama2011 = (
        region_panama + iData.posterior.sel(region_year_dim_0=7)["region_year"]
    )

    regions_years = xarray.concat(
        [
            region_year_gabon_congo2003,
            region_year_gabon_congo2005,
            region_year_gabon_congo2006,
            region_year_gabon_congo2008,
            region_year_gabon_congo2009,
            region_year_ghana2009,
            region_year_ghana2010,
            region_year_panama2011,
        ],
        dim="regions_years_dim",
    )
    regions_years.name = "region_year_2"
    iData.posterior["region_year_2"] = regions_years


def fitMultilevelB(
    virus,
    df,
    prior_predictive=False,
    target_accept=0.9,
    studentt=False,
    includeAdult=True,
    meanLifeSpanNoInfo=12,
    draws=4000,
):
    adultSuffix = "" if includeAdult else "_woAdult"
    studenttSuffix = "_studentt" if studentt else ""
    lifespanSuffix = (
        ""
        if meanLifeSpanNoInfo == 12
        else (f"_{meanLifeSpanNoInfo}_years_mean_lifespan_no_info")
    )
    outputPath = IDATA_DIR_MODEL_B / (
        f"{virus}_modelB{adultSuffix}{studenttSuffix}{lifespanSuffix}.nc"
    )

    def normalOrStudentDist(name, mu, sigma, studentt):
        return (
            pm.StudentT(name, mu=mu, sigma=sigma, nu=4)
            if studentt
            else pm.Normal(name, mu=mu, sigma=sigma)
        )

    df, dfColony = createColonysizeDf(df)
    df, dfSpecies = createSpeciesDf(df)

    assert df.sampleGroupColony.nunique() == len(dfColony)
    assert df.speciesCode.nunique() == len(dfSpecies)

    # For the solitary bats, we have an sd of 0.5 on the log scale but a maximum value
    # of 0. To solve that issue, we set the maximum value to np.log(4), allowing for
    # a maximum number of 4 animals.
    dfColony.loc[dfColony.logColonysizeMax == 0, "logColonysizeMax"] = np.log(4)

    sigma = 0.5
    sigmaBatFamily = 1
    sigmaColonysize = 0.2
    sigmaLifespan = 0.05
    sigmaRegion = 0.5
    sigmaRegionYear = 0.25
    df = df.dropna(subset=[virus, "male"])
    df["observation"] = df.index

    with pm.Model() as m:
        intercept = normalOrStudentDist("intercept", mu=0, sigma=1.5, studentt=studentt)
        b_migration = normalOrStudentDist(
            "migration", mu=0.0, sigma=sigma, studentt=studentt
        )
        b_insectivorous = normalOrStudentDist(
            "insectivorous", mu=0.0, sigma=sigma, studentt=studentt
        )
        b_cave = normalOrStudentDist("cave", mu=0.0, sigma=sigma, studentt=studentt)
        b_logColonysize = normalOrStudentDist(
            "colonysize", 0.0, sigmaColonysize, studentt=studentt
        )
        # Note that lifespan is in units of year so the expected parameter size should
        # be smaller than for the other variables.
        b_lifespan = normalOrStudentDist(
            "lifespan", 0.0, sigmaLifespan, studentt=studentt
        )
        b_male = normalOrStudentDist("male", mu=0.0, sigma=sigma, studentt=studentt)
        family_sigma = pm.HalfNormal("family_sigma", sigmaBatFamily)
        family_raw = pm.Normal("family_raw", 0.0, 1.0, shape=df.familyCode.nunique())
        a_family = pm.Deterministic("family", family_raw * family_sigma)
        a_region = pm.Normal("region", 0, sigmaRegion, shape=df.regionCode.nunique())

        # We are using a smaller standard deviation here because the year variation is
        # on top of the region variation.
        region_year_sigma = pm.HalfNormal("region_year_sigma", sigmaRegionYear)
        region_year_raw = pm.Normal(
            "region_year_raw", 0.0, 1, shape=df.regionYearCode.nunique()
        )
        a_region_year = pm.Deterministic(
            "region_year", region_year_raw * region_year_sigma
        )

        migration = pm.intX(df.migration)
        insectivorous = pm.intX(df.insectivorous)
        cave = pm.intX(df.cave)
        male = pm.intX(df.male)
        family = pm.intX(df.familyCode)
        regionYear = pm.intX(df.regionYearCode)
        region = pm.intX(df.regionCode)
        sampleGroupColony = pm.intX(df.sampleGroupColony)
        speciesCode = pm.intX(df.speciesCode)

        # The following variables are modeled assuming measurement uncertainty.
        log_colonysize_true = pm.TruncatedNormal(
            "log_colonysize_true",
            mu=dfColony.logColonysizeMean,
            sigma=dfColony.logColonysizeSd,
            lower=dfColony.logColonysizeMin,
            upper=dfColony.logColonysizeMax,
            shape=len(dfColony),
        )
        # Adding 1 to make sure that minimum lifespan is 1 year.
        lifespan_true = 1 + pm.LogNormal(
            "lifespan_true",
            mu=dfSpecies.lifespanMu,
            sigma=dfSpecies.lifespanSigma,
            shape=len(dfSpecies),
        )

        if includeAdult:
            b_adult = normalOrStudentDist(
                "adult", mu=0.0, sigma=sigma, studentt=studentt
            )
            adult = pm.intX(df.adult)
            mu = (
                intercept
                + a_region[region]
                + a_region_year[regionYear]
                + a_family[family]
                + b_migration * migration
                + b_insectivorous * insectivorous
                + b_cave * cave
                + b_logColonysize * log_colonysize_true[sampleGroupColony]
                + b_lifespan * lifespan_true[speciesCode]
                + b_adult * adult
                + b_male * male
            )
        else:
            mu = (
                intercept
                + a_region[region]
                + a_region_year[regionYear]
                + a_family[family]
                + b_migration * migration
                + b_insectivorous * insectivorous
                + b_cave * cave
                + b_logColonysize * log_colonysize_true[sampleGroupColony]
                + b_lifespan * lifespan_true[speciesCode]
                + b_male * male
            )

        p_bar = pm.Deterministic("p_bar", pm.math.invlogit(mu))

        # Bernoulli random vector with probability of success
        # given by sigmoid function and actual data as observed
        pm.Bernoulli(name="seroPos", p=p_bar, observed=df[virus])

        if prior_predictive:
            iDataCurr = pm.sample_prior_predictive(
                random_seed=RANDOM_SEED, return_inferencedata=True, var_names=["p_bar"]
            )
        else:
            if outputPath.exists():
                iDataCurr = az.from_netcdf(outputPath)
            else:
                iDataCurr = pm.sample(
                    random_seed=RANDOM_SEED,
                    target_accept=target_accept,
                    tune=2000,
                    draws=draws,
                    log_likelihood=True,
                    # nuts_sampler="numpyro",
                )
                variableTransformationModelB(iDataCurr)
                az.to_netcdf(iDataCurr, outputPath)

        return m, iDataCurr, dfColony
