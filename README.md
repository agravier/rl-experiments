# rl-experiments

Reinforcement learning notes, snippets and experiments.

I usually read Barto and Tutton's RL book when I'm in public transport or in 
other occasions when I don't have my computer, so I implement things in Python 
on my phone. I sometimes work on it on my computer too.

So this codebase is a little strange because of the limitations of my 
development environment on my phone (an Android app called 
[pydroid 3](https://play.google.com/store/apps/details?id=ru.iiec.pydroid3), a 
markdown editor and a git client).  

## Installing

You need Python 3.6 or higher. Here I assume Python 3.7.2, installed on a *nix
platform pyenv. The libraries used are listed in the requirements file, but some
that are related to visualization are platform dependent and not listed in the
requirements file.

```bash
pyenv install 3.7.2
pyenv virtualenv 3.7.2 rl-experiments
pyenv activate rl-experiments
pip install -r requirements.txt
```

Before running anything, remember to activate the virtualenv:

```bash
pyenv activate rl-experiments
```

## Testing

For practical reasons (implementing on the phone), all software tests are 
defined and run in the [tests.py](./tests.py) file.

```bash
python tests.py
```

## Running without GUI

The main.py below changes often as I go through the book and experiment with 
different things, I'll eventually make something more organized, maybe following 
the book's ToC. OR allowing to combine any compatible learner with any RL 
problem.

```bash
python main.py
```

## Graphical user interface

I mostly try to get the kivy UI to work on android. I'm not sure if it does 
anything sensible on a computer.

The kivy UI uses matplotlib to graph various parameters. Matplotlib integrates 
with kivy as follows:

- on a computer: `garden install matplotlib`
- on Android with pydroid 3: Use the quick install menu to install `kivymatplot`.

I am currently trying out some approaches in 
[kivy_mpl_test.py](rl/experiments/kivy_mpl_test.py).