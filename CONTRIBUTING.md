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


## Submitting Changes

After getting some feedback, push to your fork and submit a pull request. We
may suggest some changes or improvements or alternatives, but for small changes
your pull request should be accepted quickly.

Something that will increase the chance that your pull request is accepted:

- Write tests.
- Follow the existing coding style.
- Write a [good commit message](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html) (we follow a lowercase convention).
- For a major fix/feature make sure your PR has an issue and if it doesn't, please create one. This would help discussion with the community, and polishing ideas in case of a new feature.

## Documentation

Xplique is a small library but documentation is often a huge time sink for 
users. That's why we greatly appreciate any time spent fixing typos or 
clarifying sections in the documentation. To setup a local live-server to update
the documentation: `make serve-doc`.
 
