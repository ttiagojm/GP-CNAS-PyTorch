# GP-CNAS-PyTorch
Implementation example of "GP-CNAS: Convolutional Neural Network Architecture Search with Genetic Programming" [(Zhu et al, 2018)](https://arxiv.org/ftp/arxiv/papers/1812/1812.07611.pdf) 

***All the code is well commented explaining what each function, class or snippet of code does.***

## PyTorch version
If you want a Tensorflow version you have it <a href="https://github.com/ttiagojm/GP-CNAS">here</a>

## How to run ?
It's recommended to create a virtual environment but it's not mandatory.
- Having a CMD/Terminal inside the root directory
- Run `pip install -r requirements.txt`
- Run `pip install -e .`

This will install dependencies and execute the `setup.py`.

**Warning**: Because the simplicty of the project the constants that you can alter are available on the `main` function inside of `main.py`. There aren't any command or configuration to pass them, for now.

## Why implementing the Tree by hand ?
There are libraries to implement Binary Trees and Genetic algorithms either, but it's more flexible to implement by yourself. In this case, I didn't find any module where I can inject rules to generate a Tree. So that's the reason to implement all by hand.

## Contributions
If you find an error on the code, if something isn't well implemented or any suggestions feel free to pull request your changes and/or open [issues](https://github.com/ttiagojm/GP-CNAS/issues). Also, feel free to create a `CONTRIBUTORS.MD` and put your name and contact there.
