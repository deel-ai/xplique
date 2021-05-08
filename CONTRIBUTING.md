# Contributing

First off, thanks for taking the time to contribute! :tada::+1:

From opening a bug report to creating a pull request: every contribution is
appreciated and welcome. If you're planning to implement a new feature or change
the api please create an issue first. This way we can ensure that your precious
work is not in vain.


## Setup

- Clone the repo `git clone https://github.com/deel-ai/xplique.git`.
- Install the dependencies `make prepare-dev && source venv/bin/activate`. 
- You are ready to install the library `pip install -e .` or run the test suite
 `make test`.
 
 
## Tests

```
make test
```
In order to launch the test suite, this will run the test suite using tox on the
whole codebase for all python version. 

```
tox -e py37 -- -v test/target_file_or_folder
```
You can also pass directly through `tox` to specify arguments. For example, to run the tests on a particular module or 
file for python 3.7.


Make sure you run all the tests at least once before opening a pull request.


## Submitting Changes

After getting some feedback, push to your fork and submit a pull request. We
may suggest some changes or improvements or alternatives, but for small changes
your pull request should be accepted quickly.

Something that will increase the chance that your pull request is accepted:

- Write tests and ensure that the existing ones pass.
- Follow the existing coding style.
- Write a [good commit message](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html) (we follow a lowercase convention).
- For a major fix/feature make sure your PR has an issue and if it doesn't, please create one. This would help discussion with the community, and polishing ideas in case of a new feature.

## Documentation

Xplique is a small library but documentation is often a huge time sink for 
users. That's why we greatly appreciate any time spent fixing typos or 
clarifying sections in the documentation. To setup a local live-server to update
the documentation: `make serve-doc`.
 
