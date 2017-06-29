## Candidate Pool Machine Teaching

Candidate pool machine teaching search algorithms through a file-based
API.

## Getting Started

`capomate` provides a command-line interface to algorithms for
searching for teaching sets among a candidate pool. `capomate` is
designed to work with any learner which can be communicated with
through a file-based API.

To wit, typical usage requires from the client:

* A candidate pool of items kept in a file, one item per line
* A command which `capomate` can execute to obtain the loss of a teaching set
* Parameter settings for the search algorithm

For the details, see [Usage](#usage).

For an introduction to machine teaching,
see
[Machine Teaching: An Inverse Problem to Machine Learning and an Approach Toward Optimal Education](http://pages.cs.wisc.edu/~jerryzhu/pub/MachineTeachingAAAI15.pdf).

## Installation

Dependencies can be installed with

```
pip install numpy pandas scipy sklearn tqdm
```

This project has been tested with Python 2.7.

## Usage

    python capomate/teach.py --candidate-pool-filename CANDIDATE_POOL_FILENAME \
        --loss-executable LOSS_EXECUTABLE                                      \
        --output-filename OUTPUT_FILENAME                                      \
        --teaching-set-size TEACHING_SET_SIZE                                  \ 
        --search-budget SEARCH_BUDGET
`--candidate-pool-filename` is a file which contains the candidate pool to search from, one item per line.

`--loss-executable` is an executable which `capomate` will call during its execution. This executable must take two command-line arguments `FILE1` and `FILE2`. The first argument `FILE1` will contain a set of items for the learner to train on. The second argument `FILE2` will be a filename where the executable should write the loss of learner after training on the items in `FILE1`. The lines in `FILE1` will simply be a subset of the lines in `CANDIDATE_POOL_FILENAME`.

So for example the contents of `FILE1` might look like:

    0.0, -0.73589349999, -1.17857962828
    0.0, -0.592863350252, -0.949507866245
    0.0, -0.445851864127, -0.714059744101
    0.0, -0.277103602825, -0.443798812213
    0.0, -0.658923942945, -1.05530805171
Let's say the executable is named `my_learner`, it will be called with:

```my_learner FILE1 FILE2```

`my_learner` must train on the items in `FILE1` and write the loss of the trained learner to `FILE2` on a single line, say:

    0.03

Please note that `capomate` will use unique filenames on successive calls to the loss executable.

`--output-filename` is a filename where results are written. The first line of this file will contain the loss while the remaining lines will contain the rows out of `CANDIDATE_POOL_FILENAME` which represent the best teaching set found during search. For example, if `TEACHING_SET_SIZE` were set to 2, the output file may look something like:

    0.01
    0.0, -0.61296915225, -0.981708570076
    1.0, 1.05887652213, 1.69585721012

`--teaching-set-size` is the size of the best teaching set `capomate` will return.

`--search-budget` is the the number of models `capomate` is allowed fit before returning. This will be equivalent to the number of calls to `LOSS_EXECUTABLE`.

For example, a command-line invocation with all required parameters set might look something like:

    python capomate/teach.py                         \
        --candidate-pool-filename candidate_pool.csv \
        --loss-executable "python my_learner.py"     \
        --output-filename output.txt                 \
        --teaching-set-size 10                       \
        --search-budget 10000

## FAQ

### My learner is a MATLAB function? How can `capomate` call a MATLAB function?

One method is to wrap the call to MATLAB into a shell script. For example, let's say your MATLAB function is

    function [ output_args ] = my_learner(FILE1, FILE2)

Create the file `my_learner.sh`

    #!/bin/sh
    
    matlab -nodesktop -nosplash -nodisplay -r "my_learner $1 $2; quit" >/dev/null 2>/dev/null

Be sure to set the script's permissions with

    chmod +x my_learner.sh

And then you can call it by setting `--loss-executable ./my_learner.sh`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


