## Pool-Based Machine Teaching

Pool-based machine teaching search algorithms through a file-based
API.

## Getting Started

`poolmate` provides a command-line interface to algorithms for
searching for teaching sets among a candidate pool. `poolmate` is
designed to work with any learner which can be communicated with
through a file-based API.

To wit, typical usage requires from the client:

* A candidate pool of items kept in a file, one item per line
* A command which `poolmate` can execute to obtain the loss of a teaching set
* Parameter settings for the search algorithm

For the details, see [Usage](#usage).

For an introduction to machine teaching,
see
[Machine Teaching: An Inverse Problem to Machine Learning and an Approach Toward Optimal Education](http://pages.cs.wisc.edu/~jerryzhu/pub/MachineTeachingAAAI15.pdf).

For an overview of our research, see [here](http://pages.cs.wisc.edu/~jerryzhu/machineteaching/).

## Installation

Dependencies can be installed with

```
pip install numpy pandas scipy sklearn tqdm
```

This project has been tested with Python 2.7.

## Usage

### Command-line interface

The following describes basic usage. For more options, see [Options](#options).


    python poolmate/teach.py --candidate-pool-filename CANDIDATE_POOL_FILENAME \
        --loss-executable LOSS_EXECUTABLE                                      \
        --output-filename OUTPUT_FILENAME                                      \
        --teaching-set-size TEACHING_SET_SIZE                                  \ 
        --search-budget SEARCH_BUDGET


`--candidate-pool-filename` is a file which contains the candidate pool to search from, one item per line.

`--loss-executable` is an executable which `poolmate` will call during its execution. This executable must take two command-line arguments `FILE1` and `FILE2`. The first argument `FILE1` will contain a set of items for the learner to train on. The second argument `FILE2` will be a filename where the executable should write the loss of learner after training on the items in `FILE1`. The lines in `FILE1` will simply be a subset of the lines in `CANDIDATE_POOL_FILENAME`.

So for example the contents of `FILE1` might look like:


    1, 0.658947147839417, 0.752189242381396
    1, 0.231140742000439, -0.972920324275059
    -1, -0.830995051808994, 0.556279807173483
    -1, -0.433446335329234, -0.901179379696216
    1, 0.958199172890462, 0.286101983691191


Let's say the executable is named `my_learner`, it will be called with:

```my_learner FILE1 FILE2```

`my_learner` must train on the items in `FILE1` and write the loss of the trained learner to `FILE2` on a single line, say:


    0.03


Please note that `poolmate` will use unique filenames on successive calls to the loss executable.

`--output-filename` is a filename where results are written. The first line of this file will contain the loss while the remaining lines will contain the rows out of `CANDIDATE_POOL_FILENAME` which represent the best teaching set found during search. For example, if `TEACHING_SET_SIZE` were set to 2, the output file may look something like:


    0.00602749784937
    -1, -0.134432406608285, -0.990922765937641
    1, 0.171855154919438, -0.985122228826259


`--teaching-set-size` is the size of the best teaching set `poolmate` will return.

`--search-budget` is the the number of models `poolmate` is allowed fit before returning. This will be equivalent to the number of calls to `LOSS_EXECUTABLE`.

As an example of a command-line invocation including all required parameters, the following example runs a search over a candidate pool of items drawn uniformly from the boundary of a circle using an SVM learner:


    python poolmate/teach.py                               \
        --candidate-pool-filename poolmate/test/circle.csv \
        --loss-executable "python poolmate/test/svm.py"    \
        --output-filename output.txt                       \
        --teaching-set-size 2                              \
        --search-budget 200

Due to the stochastic nature of search, your result may differ from the loss and teaching set given in the example above.

### Programmatic Interface

If one wishes to avoid the overhead of executable callbacks and is willing to
write a learner in Python, one can invoke `poolmate` programmatically. In this
case, one has to provide a learner instance which implements two methods:

    
    class MyLearner(object):

    def loss(self, model):
        # ... return some a float loss

    def fit(self, xy):
        # ... return model fit on xy
        

The `fit` method must fit a model on `xy`, which is an iterable subset of the candidate pool.
The `loss` method receives as an argument the model returned `fit` and must itself return a loss of float type.

Here is an example of its invocation:


    from poolmate.teach import Runner, build_options
    
    runner = Runner()
    learner = MyLearner()
    options = build_options(search_budget=10000,
                            teaching_set_size=10)
    best_loss, best_set = runner.run_experiment(candidate_pool, learner, options)


## Options

Several other options are available using the command line:


    -h, --help            show this help message and exit
    --candidate-pool-filename CANDIDATE_POOL_FILENAME
                          Filename for candidate pool. The format of the file is
                          that candidate items are represented one item per
                          line. (default: None)
    --loss-executable LOSS_EXECUTABLE
                          Executable command which will return loss on teaching
                          set. Executable must take two command-line arguments,
                          an `inputfilename` containing the teaching set to
                          train the learner on, one item per line, and an
                          `outputfilename` where the loss should be written
                          (default: None)
    --output-filename OUTPUT_FILENAME
                          Output filename where the best found teaching set and
                          loss are written at the search procedure's termination
                          (default: None)
    --teaching-set-size TEACHING_SET_SIZE
                          Size of teaching set to return. (default: None)
    --search-budget SEARCH_BUDGET
                          Budget of number of models to fit. This is the number
                          of times `loss-executable` will be invoked. (default:
                          None)
    --proposals PROPOSALS
                          Number of proposals to consider at each search
                          iteration. A tuning parameter for 'greedy-add' and
                          'random-index-greedy-swap' algorithms (default: None)
    --seed SEED           Set random seed to achieve reproducibility. (default:
                          None)
    --algorithm {greedy-add,random-index-greedy-swap,uniform}
                          Choice of search algorithm (default: greedy-add)
    --initial-teaching-set INITIAL_TEACHING_SET
                          A comma-separated list of zero-based indices to fix
                          the initial teaching set. Used in 'random-index-
                          greedy-swap' and 'uniform' algorithms (e.g.,
                          --initial-teaching-set 53,17 ). (default: None)
    --log LOG             Filename of log file, where interim results are logged
                          as comma-separated values (CSV). The three colums of
                          the output represent the iteration number, the loss of
                          the trained model for that iteration, and the teaching
                          set for that iterations. (default: None)


## FAQ

### My learner is a MATLAB function? How can `poolmate` call a MATLAB function?

One method is to wrap the call to MATLAB into a shell script. For example, let's say your MATLAB function is

    function [ output_args ] = my_learner(FILE1, FILE2)

Create the file `my_learner.sh`

    #!/bin/sh
    
    matlab -nodesktop -nosplash -nodisplay -r "my_learner $1 $2; quit" >/dev/null 2>/dev/null

Be sure to set the script's permissions with

    chmod +x my_learner.sh

And then you can call it by setting `--loss-executable ./my_learner.sh`.

## Acknowledgements

This project is based upon work supported by the National Science Foundation
under Grant No. IIS-0953219. Any opinions, findings, and conclusions or
recommendations expressed in this material are those of the authors and do not
necessarily reflect the views of the National Science Foundation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

* Ara Vartanian (<aravart@cs.wisc.edu>)
* Scott Alfeld (<salfeld@amherst.edu>)
* Ayon Sen  (<ayonsn@cs.wisc.edu>)
* Jerry Zhu (<jerryzhu@cs.wisc.edu>)
