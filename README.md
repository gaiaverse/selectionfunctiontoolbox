selectionfunctiontoolbox
========================

The ``selectionfunctiontoolbox`` package provides general tools to estimate the selection functions of subsets of astronomical catalogues.
The ``selectionfunctions`` package is a product of the [Completeness of the *Gaia*-verse (CoG)](https://www.gaiaverse.space/) collaboration.

Tools in the toolbox
--------------------

Large catalogues are ubiquitous throughout astronomy, but most scientific analyses are carried out on smaller samples selected from these catalogues by carefully chosen cuts on catalogued quantities. The selection function of that scientific sample - the probability that a star in the catalogue will satisfy these cuts and so make it into the sample - is thus unique to each scientific analysis. We have created a general framework that can flexibly estimate the selection function of a sample drawn from a catalogue as a function of position, magnitude and colour. Our method is unique in using the binomial likelihood and accounting for correlations in the selection function across position, magnitude and colour using Gaussian processes and one of three different bases in the spatial dimension.

The tools we provide only differ in the basis they use to capture correlations in the selection function in the spatial dimension.

1. Hammer - uses spherical harmonics
2. Chisel - uses spherical wavelets
3. Wrench - assumes no correlation

If you have any difficulties using any of these tools, [file an issue on
GitHub](https://github.com/gaiaverse/selectionfunctiontoolbox/issues).


Installation
------------

Download the repository from [GitHub](https://github.com/gaiaverse/selectionfunctiontoolbox) and
then run:

    python setup.py install

Alternatively, you can use the Python package manager `pip`:

    pip install selectionfunctiontoolbox

Examples
--------

There are two papers associated with the ``selectionfunctiontoolbox`` package.

Boubert & Everall (2021, submitted) introduce the methodology and apply it to deduce the selection function of the APOGEE DR16 red giant sample as a subset of 2MASS. All of the code needed to reproduce the plots in that paper can be found in the Examples folder.

Everall & Boubert (2021, submitted) apply the methodology to deduce the selection functions of the astrometric and spectroscopic subsets of Gaia EDR3.

Citation
--------

If you make use of this software in a publication, please cite Boubert & Everall (2021, submitted) and Everall & Boubert (2021, submitted).