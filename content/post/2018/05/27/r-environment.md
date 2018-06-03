---
title: "My Anaconda R environment"
date: 2018-06-03
slug: "my-r-environment"
tags: ["R", "Anaconda"]
draft: false
---

## Using Jupyter Notebook with R by creating a new environment using Anaconda

In this post I want to explain the R environment that I've created. I use Anaconda (available [here](https://www.anaconda.com/download/)) for my Python work, though these steps should be the same with MiniConda. Also, I'm doing this on Windows; commands and the look of the terminal prompt will be slightly different on Mac or Linux.

### Creating a new environment

You can install the r-essentials package into your current Anaconda by running the following:

```python
    conda install -c r r-essentials
```

I prefer to create a new environment though and to do that in Anaconda, you run the following:
```python
    conda create -n r-env -c r r-essentials
```

Side note: if you don't have conda added to your PATH, you have to point to it directly:
```python
    C:\Users\username\Anaconda3\Scripts\conda create -n r-env -c r r-essentials
```

You have to explicitly say by going to the folder where Anaconda is installed. For me it's in my user folder but it could be different for you.

### Activating the environment and installing R packages

After creating the environment, you have to activate it.
```python
    activate r-env
```

Like with conda, you might have to explicit say by typing:

```python
    C:\Users\username\Anaconda3\Scripts\activate r-env
```

After activating the environment, you'll see the name of it in parentheses before the prompt, e.g. `(r-env) C:\`. Once it's activated you want to start an R terminal to install some packages. This is the message I get when I do it but you may get different.

```python
    R

    R version 3.4.3 (2017-11-30) -- "Kite-Eating Tree"
    Copyright (C) 2017 The R Foundation for Statistical Computing
    Platform: x86_64-w64-mingw32/x64 (64-bit)

    R is free software and comes with ABSOLUTELY NO WARRANTY.
    You are welcome to redistribute it under certain conditions.
    Type 'license()' or 'licence()' for distribution details.

    Natural language support but running in an English locale

    R is a collaborative project with many contributors.
    Type 'contributors()' for more information and
    'citation()' on how to cite R or R packages in publications.

    Type 'demo()' for some demos, 'help()' for on-line help, or
    'help.start()' for an HTML browser interface to help.
    Type 'q()' to quit R.

    Microsoft R Open 3.4.3
    The enhanced R distribution from Microsoft
    Microsoft packages Copyright (C) 2017 Microsoft Corporation

    Using the Intel MKL for parallel mathematical computing (using 6 cores).

    Default CRAN mirror snapshot taken on 2017-09-01.
    See: https://mran.microsoft.com/.

    >
```

To install the packages, run the following:

```python
    > install.packages(c('repr', 'IRdisplay', 'evaluate', 'crayon', 'pbdZMQ', 'devtools', 'uuid', 'digest'))
```

You will most likely be prompted to select a CRAN mirror for package installation. After doing so and they successfully install, you need to install IRkernel.

```python
    > devtools::install_github('IRkernel/IRkernel')
```

Once that installs, you need to make the R kernel visible to Jupyter Notebook by running the following:
```python
    > # For the current user only
    > IRkernel::installspec()
    > # For all users
    > IRkernel::installspec(user=FALSE)
```

Now when you start Jupyter you'll see R appearing in the list of kernels when creating a new notebook. You can start Jupyter in this environment just like you would normally, by typing `jupyter notebook` and hitting Enter.

### Deactivating the environment

Once you are done installing packages, quit the R terminal by typing `q()` and hitting Enter.

```python
    > q()
```

It'll ask if you want to save the workspace image and you can hit `n`. Then deactivate the environment just by typing `deactivate`. You'll know it's deactivated because the environment name will no longer be in parentheses before the prompt, e.g. `C:\`.

```python
    deactivate
```

Or

```python
    C:\Users\username\Anaconda3\Scripts\deactivate
```

That's it. I find this environment and Anaconda environments in general to be really helpful. I've created ones for web scraping, web development with Flask, and will create more based on what I'm doing. That's all for this post!