#! /usr/bin/env python

from utils.dataUtils import loadData, preprocessDfModelSimple, preprocessDfModelB
from utils.plotUtils import returnViruses
from utils.regression import fitSimple, fitMultilevelB


def runModelSimple(df, virus, studentt=False):
    dfSimple = preprocessDfModelSimple(df)

    print(f"Processing virus {virus}")
    m, iData = fitSimple(virus, dfSimple, target_accept=0.9, studentt=studentt)

    return m, iData


def runModelB(df, virus, includeAdult=True, studentt=False):
    dfMultilevelB = preprocessDfModelB(df)

    print(f"Processing virus {virus}")
    if virus in ("rvfv", "sars-cov", "chikv", "veev"):
        # Models for these viruses showed divergences with lower target_accept values.
        target_accept = 0.98
    else:
        target_accept = 0.95
    m, iData, _ = fitMultilevelB(
        virus,
        dfMultilevelB,
        target_accept=target_accept,
        studentt=studentt,
        includeAdult=includeAdult,
    )
    return m, iData


def runRegression(df, model="A"):
    allViruses = returnViruses()
    for virus in allViruses:
        if model == "A":
            _, _ = runModelSimple(df, virus)
        else:
            _, _ = runModelB(df, virus)


if __name__ == "__main__":
    dfOrig, df, dfTropical = loadData()
    runRegression(dfTropical, model="A")
    runRegression(dfTropical, model="B")
