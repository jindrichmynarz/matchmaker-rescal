# matchmaker-rescal

Evaluation runner for the [RESCAL](https://github.com/mnick/rescal.py)-based matchmakers.

## Installation

```bash
pip install .
pip install -r requirements.txt
```

## Usage

Observe the command-line parameters:

```bash
python matchmaker_rescal/cli.py -h
```

There are the following parameters:

* `-g/--ground-truth` (required): Path to [MatrixMarket](http://math.nist.gov/MatrixMarket/formats.html#MMformat) file with the matrix that contains the ground truth about relations between contracts and their awarded bidders.
* `-s/--slices` (optional): Paths to [MatrixMarket](http://math.nist.gov/MatrixMarket/formats.html#MMformat) files with matrices that represent slices of a tensor factorized by RESCAL. You can provide an arbitrary number of slices.
* `-c/--config` (optional): Path to a configuration file in [EDN](https://github.com/edn-format/edn). You can see default configuration [here](https://github.com/jindrichmynarz/matchmaker-rescal/blob/master/matchmaker_rescal/resources/config.edn). 

The input matrices for the ground truth and the slices can be produced by [sparql-to-tensor](https://github.com/jindrichmynarz/sparql-to-tensor).
